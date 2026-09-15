# Boundary lines emit as one LWPOLYLINE per traced run, not one LINE per call

docproc's DXF export research (`docs/research/dxf-autocad-export.md` in docproc)
recommends emitting each reviewed straight boundary call as its own `LINE`
entity, with `LWPOLYLINE` only optional for an already-connected all-straight
boundary. That recommendation fits docproc's accepted model: geometry is built
from human-reviewed calls, one call at a time.

vectorjuju's pipeline is the opposite. Geometry comes from traced raster
linework (skeletonize -> trace -> classify) first; parsed boundary calls are
bound to that geometry afterward as labels, never used to reconstruct it (see
the map's Notes on [Map: vectorjuju convert() — survey raster/PDF to DXF
MVP](https://github.com/monocongo/vectorjuju/issues/1)). docproc's own
prototype implementation plan already reached the same conclusion for this
architecture: traced lines emit as `LWPOLYLINE`, "not dense polylines."

**Decision:** `convert()` emits each traced straight-line run as one
`LWPOLYLINE` (closed when the run is a closed boundary loop), not one `LINE`
per parsed call. A call's bearing/distance is not recoverable from the DXF
geometry alone — it lives in the JSON sidecar's `entities[].label` instead.

Settled in [Define the convert() and DXF output
contract](https://github.com/monocongo/vectorjuju/issues/5).
