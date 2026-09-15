> Carried from docproc [`docs/research/dxf-autocad-export.md`](https://github.com/monocongo/docproc/blob/d9ae59a2cafee088fb03b56c37cc92f03b33875b/docs/research/dxf-autocad-export.md) at `d9ae59a2cafee088fb03b56c37cc92f03b33875b`.
>
> Written for docproc's persisted, review-required application. vectorjuju's `convert()` is a stateless function with no persistence, review workflow, or UI (see the map's Out of scope) — treat every mention below of a "Survey Review Revision", `DOCPROC` APPID, or review-required delivery notice as docproc-only context, not applicable here. The DXF header/units, entity (`LINE`/`ARC`/`POINT`/`TEXT`/`LWPOLYLINE`), and validation conventions still apply.

# AutoCAD-compatible DXF export conventions

**Issue:** [#105](https://github.com/monocongo/docproc/issues/105)

**Scope:** public format and library research for a local, review-required drafting export. No plat or other source document was accessed, copied, uploaded, or used for this research.

## Answer

**Recommendation (requires product/architecture review):** export an ASCII **AutoCAD 2000 DXF** (`$ACADVER = AC1015`) with explicit `$INSUNITS`, planar local WCS coordinates, a small defined layer table, and only `LINE`, `ARC`, `POINT`, `TEXT`, and—where a contiguous all-straight boundary is useful—`LWPOLYLINE`. Use a pinned [ezdxf](https://github.com/mozman/ezdxf) release to write it. Keep the DXF a review-required drafting Artifact: the immutable Survey Review Revision and detailed source-region evidence remain authoritative in the application, not in a mutable CAD file.

**Policy recommendation:** `AC1015` is the practical compatibility floor: it supports `LWPOLYLINE`, while avoiding a newer version unless native AutoCAD geolocation is actually required. If the reviewed export must carry AutoCAD-native geolocation, use **AutoCAD 2010** (`AC1024`) and a reviewed `GEODATA` object instead; do not silently upgrade or invent a CRS.[^header][^lwpolyline][^geodata]

## Format facts (normative source semantics)

| Concern | Autodesk/official-project fact |
| --- | --- |
| Version and units | `$ACADVER` identifies the drawing database version; `AC1015` is AutoCAD 2000 and `AC1024` is AutoCAD 2010. `$INSUNITS` records default drawing units for DesignCenter blocks: `2` feet, `6` metres, and `21` US survey feet.[^header] |
| Coordinates | `LINE` uses WCS points. `ARC` and `LWPOLYLINE` use OCS; their extrusion direction defaults to `(0, 0, 1)`. An `ARC` stores OCS centre, radius, start angle, and end angle; `LWPOLYLINE` vertices are OCS coordinates and its closed flag is bit 1.[^coords][^arc][^lwpolyline] |
| Layers | A graphical entity's group code `8` names its layer. `LAYER` table records define the name, flags, colour, and linetype.[^entity-codes][^layer] |
| Entity metadata | XDATA follows an entity's normal data. Each application group begins with group code `1001`, whose application name corresponds to an `APPID` table entry; string values use group code `1000`.[^xdata][^appid] |
| Native geolocation | `GEODATA` includes a coordinate type, WCS design point, north direction, horizontal/vertical unit scales, and a coordinate-system definition string. ezdxf documents `GEODATA` as requiring DXF R2010 / `AC1024`.[^geodata][^ezdxf-geodata] |
| Writer and license | ezdxf's official repository documents DXF support including R2000 and newer releases. Its MIT license permits use, copying, modification, and distribution when the copyright and permission notice are retained, and disclaims warranty.[^ezdxf-readme][^ezdxf-license] |
| Native validation | AutoCAD `AUDIT` evaluates drawing integrity; with `AUDITCTL=1`, it writes an `.adt` report of problems and actions. It can ask whether to repair errors.[^audit] |

## Recommended minimal export contract

These are **recommendations**, not claims that the DXF specification requires the policy.

1. **Header and plane.** Write `AC1015`, set `$INSUNITS` to the reviewed source unit (`2`, `6`, or `21`; never infer feet versus US survey feet), and use one local WCS XY plane with `z = 0`. Set a known insertion base and truthful WCS extents. Do not depend on a saved UCS, page-image position, or a silent grid/ground conversion.
2. **Coordinate declaration.** For a local coordinate basis, record a compact reviewed declaration: basis, datum/CRS identifier when known, unit, grid-or-ground basis, and any confirmed transform identifier. This is descriptive metadata, not a native AutoCAD geolocation claim. Use `AC1024` plus `GEODATA` only when a reviewer has explicitly supplied and approved the native geolocation inputs.
3. **Geometry.** Emit each reviewed straight boundary call as `LINE` and each reviewed circular call as a true `ARC`; retain the reviewed centre, radius, and endpoints/angles rather than chord-approximating a curve. With the default normal, calculate and validate its degree angles and counter-clockwise start-to-end direction. Use `POINT` for shared control or monument points and `TEXT` only for display labels. `LWPOLYLINE` is optional for an all-straight connected boundary; set it closed only when the reviewed geometry is closed. Do not use bulges for reviewed circular calls in the first contract, and reject/flag unsupported non-circular curves rather than approximating them.[^ezdxf-arc]
4. **Layers.** Define and use exactly `SURVEY_BOUNDARY`, `SURVEY_CONTROL`, `SURVEY_LABEL`, and `DOCPROC_METADATA`; use `CONTINUOUS` and ByLayer graphics. The first three carry the corresponding entities. The metadata layer contains one visible `TEXT` review-required notice, which is also the metadata anchor. It is an advisory delivery notice, not a substitute for the application's review authority. Do not export unrelated plat features, hatches, dimensions, blocks, proxy objects, or a general CAD-editing model.
5. **Metadata and evidence.** Register one `DOCPROC` APPID. Attach compact XDATA to every exported geometry entity with at least its stable reviewed feature identifier and role. Attach drawing-level fields—`review_required`, reviewed units/basis/datum, source-document digest(s), Survey Review Revision identifier, generation time, and acknowledged closure warning(s)—to the `TEXT` metadata anchor. Keep each XDATA string within 255 bytes and the anchor's total XDATA within 16 KB; reject an export that exceeds either limit rather than dropping provenance. Retain detailed source regions, observations, and immutable evidence only in the application. XDATA and a CAD layer are mutable delivery metadata; they neither replace nor alter docproc's immutable Artifact and Evidence Record contracts.[^xdata][^ezdxf-xdata][^evidence-contract]
6. **Python library and license.** Use ezdxf rather than hand-writing DXF group codes. Pin the chosen release in the eventual implementation, retain the required MIT notice in applicable third-party-license material, and record the writer version as export provenance. This research adds neither the dependency nor production code.

## Validation: separate writer checks from independent acceptance

1. **Writer preflight (not independent):** serialize, reopen with ezdxf, and run its documented audit/validation facilities. Treat any reported error or automatic repair as an export failure rather than silently shipping the repaired file.[^ezdxf-audit]
2. **Independent AutoCAD check:** on a disposable local copy, set `AUDITCTL=1`, run AutoCAD `AUDIT`, initially decline repair, and retain the `.adt` report. Any finding or required repair fails this contract. This is the primary compatibility gate because AutoCAD is a separate implementation from the writer.[^audit]
3. **Independent semantic comparison:** use a small DXF group-code-pair decoder that neither imports ezdxf nor shares the exporter's geometry conversion, then compare its result against a deterministic manifest from the approved Survey Review Revision: header version and units; layer/entity counts; feature-ID set; point/line endpoints; arc centre/radius/start/end-derived endpoints; extents; and declared closure status, all under an explicit tolerance. This detects a syntactically valid but semantically wrong export; AutoCAD `AUDIT` alone does not provide this comparison.
4. **Optional round trip:** save the disposable AutoCAD-opened copy and repeat the semantic comparison. Review every geometry or delivery-metadata change; successful opening alone is not acceptance.

The checks operate locally. They do not require uploading a DXF or any source document to a web service.

## Sources

[^header]: Autodesk, [HEADER section group codes](https://help.autodesk.com/cloudhelp/2025/ENU/AutoCAD-DXF/files/GUID-A85E8E67-27CD-4C59-BE61-4DC9FADBE74A.htm).
[^coords]: Autodesk, [About the Object Coordinate System (OCS)](https://help.autodesk.com/cloudhelp/2025/ENU/AutoCAD-DXF/files/GUID-D99F1509-E4E4-47A3-8691-92EA07DC88F5.htm); [LINE](https://help.autodesk.com/cloudhelp/2025/ENU/AutoCAD-DXF/files/GUID-FCEF5726-53AE-4C43-B4EA-C84EB8686A66.htm).
[^arc]: Autodesk, [ARC](https://help.autodesk.com/cloudhelp/2025/ENU/AutoCAD-DXF/files/GUID-0B14D8F1-0EBA-44BF-9108-57D8CE614BC8.htm).
[^ezdxf-arc]: ezdxf, [ARC](https://ezdxf.readthedocs.io/en/stable/dxfentities/arc.html).
[^lwpolyline]: Autodesk, [LWPOLYLINE](https://help.autodesk.com/cloudhelp/2025/ENU/AutoCAD-DXF/files/GUID-748FC305-F3F2-4F74-825A-61F04D757A50.htm); ezdxf, [LWPOLYLINE](https://ezdxf.readthedocs.io/en/stable/dxfentities/lwpolyline.html).
[^entity-codes]: Autodesk, [Common group codes for entities](https://help.autodesk.com/cloudhelp/2025/ENU/AutoCAD-DXF/files/GUID-3610039E-27D1-4E23-B6D3-7E60B22BB5BD.htm).
[^layer]: Autodesk, [LAYER](https://help.autodesk.com/cloudhelp/2025/ENU/AutoCAD-DXF/files/GUID-D94802B0-8BE8-4AC9-8054-17197688AFDB.htm).
[^xdata]: Autodesk, [Extended Data (XDATA)](https://help.autodesk.com/cloudhelp/2025/ENU/AutoCAD-DXF/files/GUID-A2A628B0-3699-4740-A215-C560E7242F63.htm).
[^appid]: Autodesk, [APPID](https://help.autodesk.com/cloudhelp/2025/ENU/AutoCAD-DXF/files/GUID-6E3140E9-E560-4C77-904E-480382F0553E.htm).
[^ezdxf-xdata]: ezdxf, [Custom data: XDATA limits](https://ezdxf.readthedocs.io/en/stable/tutorials/custom_data.html).
[^geodata]: Autodesk, [GEODATA](https://help.autodesk.com/cloudhelp/2025/ENU/AutoCAD-DXF/files/GUID-104FE0E2-4801-4AC8-B92C-1DDF5AC7AB64.htm).
[^ezdxf-geodata]: ezdxf, [GEODATA](https://ezdxf.readthedocs.io/en/stable/dxfobjects/geodata.html).
[^ezdxf-readme]: ezdxf official repository, [README](https://github.com/mozman/ezdxf/blob/master/README.md).
[^ezdxf-license]: ezdxf official repository, [MIT license](https://github.com/mozman/ezdxf/blob/master/LICENSE).
[^audit]: Autodesk, [AUDIT](https://help.autodesk.com/cloudhelp/2025/ENU/AutoCAD-Core/files/GUID-62DDB935-61B1-49DA-8238-3EF1CC45259B.htm).
[^ezdxf-audit]: ezdxf, [Drawing audit and validation](https://ezdxf.readthedocs.io/en/stable/drawing/drawing.html).
[^evidence-contract]: docproc, [Identity invariants and evidence content-addressing contract](https://github.com/monocongo/docproc/blob/d9ae59a2cafee088fb03b56c37cc92f03b33875b/docs/decisions/evidence-content-addressing.md).
