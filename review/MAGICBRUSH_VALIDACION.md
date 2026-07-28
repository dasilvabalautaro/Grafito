# Validación del editor local MagicBrush

Registro completo de las pruebas hechas al checkpoint `modelos/grafito-magicbrush-v2`
antes y después de integrarlo como motor de edición local. Todo lo que aparece aquí
se ejecutó en el equipo de desarrollo; no hay cifras estimadas ni heredadas de la
documentación del modelo.

## Equipo y configuración

| | |
|---|---|
| Equipo | iMac Intel · Radeon Pro 5500 XT (presupuesto MPS ≈ 13,6 GB) |
| Ejecución | MPS, `float16` |
| Software | PyTorch 2.2.2 · Diffusers 0.32.2 · Transformers 4.48.3 |
| Modelo | InstructPix2Pix afinado con MagicBrush, base SD 1.5 |
| Pesos | 3,6 GB en Safetensors fp32 (UNet 3,44 GB ≈ 0,86 B parámetros) → ≈ 1,7 GB en fp16 |
| UNet | `in_channels: 8` (4 latentes + 4 de la imagen), `cross_attention_dim: 768`, `sample_size: 64` |
| Scheduler | `EulerAncestralDiscreteScheduler` (el del propio checkpoint) |

Detalle de carga: el `model_index.json` declara un `safety_checker` cuya carpeta no
existe en el paquete. Sin `safety_checker=None, requires_safety_checker=False` la
carga falla; con esos argumentos, funciona a la primera.

## Imagen de prueba

Todas las rondas usan el mismo retrato real,
`output/flux-editada-20260722-080828-3de40425.png` (1024×688 px): una persona con
gorra roja, gafas y chaqueta oscura, en un interior con estanterías, plantas y un
árbol pintado en la pared. Se eligió a propósito porque concentra los cuatro
elementos que un editor por instrucciones suele romper: **rostro, gafas, gorra y un
fondo con detalle**.

Al procesarse queda en **512×344 px**: el lado mayor se ajusta a 512 conservando la
relación de aspecto y ambos lados se redondean a múltiplos de 8.

## Metodología

El filtro es el que ya estaba fijado en
[LOCAL_IMAGE_GENERATION.md](LOCAL_IMAGE_GENERATION.md), punto 4 y 5: cinco órdenes
—**añadir, quitar, reemplazar, cambiar color/material y cambiar fondo**— sin ajustes
específicos por objeto, y aprobación solo si conserva rostro y fondo y cumple al
menos cuatro de cinco. Semilla fija en 0 salvo donde se indique, para que las
comparaciones entre rondas sean directas.

---

## Ronda 1 · Primera edición, comprobación de que el pipeline funciona

Parámetros: guía de texto 7,5 · fidelidad a la imagen 1,5 · 20 pasos · sin recorte de
bordes.

| Orden | Resultado |
|---|---|
| `pon un sombrero rojo` (en español) | Sombrero rojo de ala ancha, tipo mexicano. Identidad, gafas, ropa y fondo intactos |

**Hallazgo importante y no previsto:** la orden iba en español y el modelo respondió,
pero interpretó «sombrero» de forma literal como el sombrero mexicano en lugar de un
gorro genérico. **El modelo se entrenó con instrucciones en inglés**; en español
acierta de forma imprevisible. De aquí salieron el aviso de la interfaz y los
ejemplos en inglés.

Tiempo: **43,0 s** en proceso frío (incluye 4 s de carga del pipeline y el
calentamiento de kernels de Metal).

## Ronda 2 · Filtro de cinco órdenes

Mismos parámetros que la ronda 1, ahora en inglés y en un solo proceso.

| Orden | Instrucción | Resultado |
|---|---|---|
| Añadir | `add a pair of headphones on his head` | ✅ Auriculares puestos y bien integrados. Gorra, gafas, rostro y fondo intactos |
| Quitar | `remove his glasses` | ❌ **Falló.** Quitó las gafas, pero también la gorra, y deformó la zona de los ojos: párpados cerrados y frente alisada |
| Reemplazar | `replace the red cap with a blue beanie` | ✅ Gorro azul en lugar de la gorra roja. Gafas, rostro y fondo intactos |
| Color / material | `make his jacket bright yellow` | ✅ Chaqueta amarilla. Rostro, gorra, gafas y fondo intactos. El cuello cambia algo de forma |
| Fondo | `change the background to a sandy beach` | ✅ Fondo sustituido por playa. Sujeto completamente preservado. La arena queda tosca en los bordes |

**Resultado: 4 de 5**, con rostro y fondo preservados en los cuatro éxitos. Cumple el
criterio de aceptación (≥ 4/5). Motor aprobado.

**Limitación medida: quitar objetos es su punto débil.** No es un problema de ajuste:
la orden de eliminación arrastra objetos vecinos y deforma la región. Para eliminar
algo, FLUX API.

## Ronda 3 · Comparación con los parámetros del servicio original

`prm_magic.txt` —el `api_server.py` del proyecto que entrenó el checkpoint— propone
valores distintos. Se probaron todos.

Parámetros de esta ronda: guía de texto **7,0** · attention slicing **activo** ·
recorte de bordes 8 px.

| Prueba | Instrucción | Resultado |
|---|---|---|
| Genérico | `add a hat` | Gorra blanca y roja tipo trucker. Identidad y fondo intactos |
| Concreto | `add a black hat` | Gorra negra. **Obedece el color pedido**; identidad y fondo intactos |
| Semilla aleatoria | `add a black hat`, semilla −1 | Funciona: eligió la 1866857604 y la informó de vuelta |

**Confirmado el hallazgo del autor del checkpoint:** concretar el objeto
(`add a black hat`) integra mejor que la orden genérica (`add a hat`), que improvisa
el color y el tipo de prenda.

Tiempos: 32 s la primera edición del proceso, **23 s** las siguientes.

### Veredicto sobre cada parámetro del servicio original

| Ajuste propuesto | Decisión | Motivo |
|---|---|---|
| Recorte de 8 px por lado | **Adoptado** | El checkpoint ensucia el borde exterior con manchas de esquina esporádicas |
| `guidance_scale` 7,0 | **Descartado** | 7,5 dio mejores resultados en la comparación directa |
| Attention slicing siempre fuera de CUDA | **Descartado** | Resultados peores aquí. Queda disponible con `INSTRUCT_LOCAL_ATTENTION_SLICING=auto` |
| Semilla negativa = aleatoria | **Adoptado** | No afecta a la imagen y permite explorar variantes |
| Barras de progreso silenciadas | **Adoptado** | No afecta a la imagen |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` | **No adoptado por defecto** | Es una variable de proceso: también cambiaría el perfil de memoria de SDXL, que está validado con el techo puesto. Disponible en `INSTRUCT_LOCAL_MPS_UNLIMITED=true` |
| Respaldo a `timbrooks/instruct-pix2pix` si falta el checkpoint | **No adoptado** | Ese modelo base es justo el que el proyecto ya había descartado por deformar. Aquí falta el modelo y se dice, en vez de degradar sin avisar |

## Ronda 4 · Validación final con los parámetros definitivos

Parámetros servidos hoy: guía de texto **7,5** · fidelidad a la imagen **1,5** ·
**20 pasos** · 512 px · attention slicing **desactivado** · recorte de bordes **8 px**
· semilla 0.

Las siete imágenes se conservan en `output/`:

| Archivo | Instrucción | Resultado |
|---|---|---|
| `magicbrush-validacion-anadir.png` | `add a pair of headphones on his head` | ✅ |
| `magicbrush-validacion-quitar.png` | `remove his glasses` | ❌ se lleva la gorra y deforma los ojos |
| `magicbrush-validacion-reemplazar.png` | `replace the red cap with a blue beanie` | ✅ |
| `magicbrush-validacion-color.png` | `make his jacket bright yellow` | ✅ |
| `magicbrush-validacion-fondo.png` | `change the background to a sandy beach` | ✅ |
| `magicbrush-validacion-sombrero-generico.png` | `add a hat` | Gorra genérica, color improvisado |
| `magicbrush-validacion-sombrero-concreto.png` | `add a black hat` | Gorra negra, obedece el color |

**Los resultados coinciden con los de la ronda 2**, lo que confirma que volver a 7,5
y desactivar el slicing restauró el comportamiento anterior.

## Tiempos medidos

| Momento | Tiempo |
|---|---|
| Carga del pipeline | ~4 s |
| Primera edición del proceso (carga + calentamiento incluidos) | ~28-43 s |
| Ediciones siguientes, en caliente | **~23 s** |

A 512×344 px y 20 pasos. Para contexto: son ~1,1 s por paso, frente a los 3,76 s por
paso que cuesta el UNet de SDXL en el mismo equipo.

## Conclusiones

1. **El motor está aprobado**: 4 de 5 órdenes con identidad y fondo preservados, sin
   ajustes específicos por objeto.
2. **No sirve para quitar objetos.** Es el fallo reproducible de las tres rondas.
3. **Las órdenes van en inglés**, breves y concretas. Nombrar color o material mejora
   la integración.
4. **La instrucción se envía literal.** No pasa por Qwen ni recibe prefijos de estilo:
   expandirla a una descripción de escena hace que el modelo repinte lo que debía
   conservar. Por eso los controles de Qwen se ocultan al elegir este motor.
5. **De los parámetros del servicio original solo sobrevivió el recorte de bordes.**
   Los valores por defecto están fijados por
   `test_magicbrush_keeps_the_defaults_validated_on_this_machine` para que no se
   realineen con ese servicio sin repetir la comparación.

## Límites conocidos, para vigilar en futuras pruebas

- Quitar objetos (medido aquí).
- Cambios de estilo global: irregulares; el autor advierte que no es su fuerte.
- Regiones muy grandes y texto legible.
- Editar el fondo puede teñir la ropa si la adherencia sale floja.
- Los interiores aclarados (`make the room brighter`) tienden a un tinte magenta.
- Manchas de esquina: ya mitigadas con el recorte automático de 8 px.

## Cómo reproducir

```bash
source .venv/bin/activate
python scripts/check_magicbrush.py output/<foto>.png "add a black hat"
python scripts/check_magicbrush.py output/<foto>.png "remove his glasses" --image-guidance 2.0
```

El guion imprime dispositivo, dtype, resolución de trabajo, tiempo, semilla usada y
ruta del resultado, sin levantar Gradio ni gastar créditos.
