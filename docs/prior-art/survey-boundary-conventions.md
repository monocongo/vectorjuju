> Carried from docproc [`docs/research/survey-boundary-conventions.md`](https://github.com/monocongo/docproc/blob/12bede5afdf4656412c82c72938c2cbcbe11e2a3/docs/research/survey-boundary-conventions.md) at `12bede5afdf4656412c82c72938c2cbcbe11e2a3`.

# U.S. survey boundary-call and traverse conventions for an initial North Carolina model

**Scope.** This is a source map for preserving reported survey semantics, not a
survey standard, a finding that a particular document complies, or legal advice.
North Carolina requirements apply only in their stated jurisdictional and
survey/recording scope; a survey's date, purpose, and local requirements can
matter. Sources were retrieved 2026-09-10. No non-authoritative explainers were
used.

## Answer: source hierarchy

| Concern | Controlling North Carolina source | Computational/semantic authority | What the model must not lose |
| --- | --- | --- | --- |
| Direction calls | Recorded plats must disclose the north index and show each surveyed property's course or azimuth and distance. The north index can be true, magnetic, North Carolina grid, or old deed/plat bearings; grid requires datum and realization, while magnetic/record bearings require the original date and source when known. [S1, § 47-30(f)(1)–(2)] | FGDC distinguishes a quadrant bearing from a full-circle azimuth and identifies the basis of direction as separate data. [S6, pp. 59, 69] | Reported bearing/azimuth notation; N/S and E/W quadrant letters or azimuth sweep; the north/basis statement; datum and realization where stated; source/date for magnetic or record bearings. |
| Linear units | A recorded plat's surveyed-line distances are in U.S. Survey feet or metres. [S1, § 47-30(f)(2)] | NIST says the U.S. survey foot is obsolete for new work from 2023 and the international foot is exactly 0.3048 m; it remains relevant to historical/legacy material. [S9] | Original unit spelling and precision, especially **U.S. Survey foot** versus foot/international foot versus metre; never silently rewrite a reported unit. |
| Straight and curved boundaries | Every surveyed property line has a direction and distance. A curved boundary must report actual PC-to-PT curve data as standard curve data or a traverse; standard data includes the long-chord bearing and distance. [S1, § 47-30(f)(2), (4)] | FGDC defines straight line, circular curve, radius, central angle, curve direction, degree-of-curve type, long chord, and arc length. [S6, pp. 23, 59, 69] | Whether the record calls a straight line, circular curve, or other curve; PC/PT and every reported curve element; whether it reports standard data, a traverse, or subchords. Missing elements stay missing. |
| Monuments and control | Marked monument/natural-object corners must be identified. Board rules require artificial monuments to be described as found or set and define ties to nearby government control and their reproduction purpose. [S1, § 47-30(f)(6), (9); S2, Rules .1602(f)–(g), .1604(d)(6), (9)] | NGS is the source incorporated by the Board rule for geodetic datums. [S2, Rule .1602(g)] | Monument identity, found/set status, reported description, control-station identity, tie calls, and the source of the tie—not merely a calculated coordinate. |
| Ground/grid observations | Plat distances are horizontal ground or horizontal grid; North Carolina grid use requires the combined grid factor and grid distances must be indicated. [S1, § 47-30(f)(3)] | FGDC treats direction basis and distance reference surface (ground, sea-level/geodetic, or grid) as independent attributes. [S6, p. 59] | Ground/grid basis, factor as reported, and whether a distance is observed/reported or derived. A factor must not be inferred from coordinates. |
| CRS and coordinates | When a surveyed plat shows coordinates, they are X/easting and Y/northing, traceable to a published geodetic datum or NC State Plane; the datum, realization, and establishment method/data are to be shown. [S1, § 47-30(f)(9); S2, Rules .1602(g), .1604(c)–(d)(1)] | NGS defines State Plane as conformal map projections: a projection is a mathematical conversion of latitude/longitude to plane northing/easting. [S7] | Coordinate axes/order, coordinate system, datum/reference frame, realization, epoch when stated, horizontal versus vertical datum, control source, and establishment method. `NAD 83` alone is not an interchangeable realization label. |
| Closure and quality | A recorded plat reports its ratio of precision **or** positional accuracy before adjustments. Board classes set angular-closure, precision-ratio, and 95%-confidence positional-accuracy limits; a subdivision perimeter additionally needs continuous closure. [S1, § 47-30(d), (f)(5); S2, Rules .1603, .1604(d)(5), (12)] | — | The reported quality kind (ratio, angular closure, or positional accuracy), value, class, confidence level, reference control, and explicit before-adjustment status. Do not reduce these to one generic “closure” number. |
| Conversions and transformations | When plat bearings and shown grid coordinates use different references, the plat must provide either a second tied coordinate or written and graphical conversion information. [S1, § 47-30(f)(9)] | NGS distinguishes coordinate-system conversion from transformation between reference frames/datums; NCAT uses NADCON for 3-D coordinate transformations and VERTCON for orthometric-height transformations. [S8] | Source and target CRS/frame/realization/epoch, operation type, named grid/model/factor and version when reported, and derived output separately from the reported call. |

## North Carolina requirements to preserve

### Recordable plats and professional practice

For a plat within the scope of G.S. 47-30, the statutory disclosures above are
the first preservation baseline. In particular, a model cannot safely collapse:

- a quadrant bearing into an unlabelled numeric azimuth;
- U.S. Survey feet into an unqualified “feet” value;
- horizontal ground and grid distance into one distance kind;
- a monument, control point, and calculated corner into one generic point; or
- a pre-adjustment precision ratio, angular closure, and positional accuracy
  into one accuracy field. [S1, § 47-30(d), (f)]

The Board-published rule compilation used here says the standards of practice
apply to Professional Land Surveyors in land surveying. It adds the map/report
and tie-line requirements, the boundary classes, and GNSS metadata. For GNSS
work, retain the reported datum **and epoch**, fixed control and position, geoid
model, combined grid factor, units, and positional accuracy; fixed-station
presentation also identifies height type and datum/epoch. [S2, Rules
.1601–.1604, .1607(b), (e)]

The cited compilation records amendments through 2020 for the relevant rules.
Confirm the then-current official administrative-code text before treating a
requirement as current legal compliance advice.

### Coordinate-system statutes

The coordinate-system statutes are source semantics, not permission to replace
the system named on a record:

- G.S. 102-1.1 describes the North Carolina Coordinate System of 1983 as a
  Lambert conformal projection on GRS 80 with metre axes and a U.S. survey-foot
  conversion factor. [S3]
- G.S. 102-1.2 describes the North Carolina Coordinate System of 2022 on
  NATRF2022, with metre axes and the international foot exactly 0.3048 m, but
  makes it operative only after the specified NCGS receipt of NGS notice of a
  complete published definition. [S4]
- G.S. 102-7 says Chapter 102 does not require a purchaser or mortgagee to
  rely wholly on a North Carolina Coordinate System description. [S5]

Therefore retain the stated system/frame and its realization rather than
assuming a record is in the newest named system or that coordinates displace the
boundary evidence.

## Computational conventions, not independent legal requirements

FGDC's *Cadastral Data Content Standard* is the appropriate federal semantic
vocabulary for a typed geometry record: it supplies distinct concepts for
straight/circular/other boundary, bearing/azimuth and direction basis, distance
unit/basis, and curve components. It expressly does **not** prescribe land
survey procedures; private and non-federal public land surveys follow governing
state law. Use it to name and retain data, not to decide North Carolina
compliance. [S6, §§ 1.4, 3.2]

Similarly, NGS's State Plane and NCAT material is authoritative for the
mathematics and operations: map-projection conversion is not the same operation
as reference-frame/datum transformation. It does not turn a calculated
conversion into the original legal call or select a North Carolina plat's
applicable standard. [S7; S8]

NIST's national unit guidance does not authorize changing a North Carolina
record that explicitly reports U.S. Survey feet. Preserve the declaration first;
any international-foot or SI result is a separately identified derived value.
[S1, § 47-30(f)(2); S9]

## Minimal preservation baseline

This is a record-content checklist, **not** a proposed production schema:

1. Keep the exact reported call and source-region evidence alongside any parsed
   value. A canonical direction, coordinate, or geometry is derived data and
   must not overwrite the reported call.
2. For each straight call, retain direction form, components, north/basis,
   distance, unit, and ground/grid basis. For each curve, retain its reported
   representation and PC/PT, long chord, radius, arc length, central angle,
   direction, degree definition, or subchords only when present.
3. Keep point evidence separate: corner role; monument/natural-object identity;
   found/set/unknown status; control station; and tie line. A control survey
   itself cannot define or convey rights or ownership. [S1, § 47-30(f)(11)c.3]
4. Attach CRS metadata to coordinates and directional/grid context: axis labels,
   datum/reference frame, realization, epoch, projection/system, control and
   establishment method. Preserve vertical datum, height type, and geoid model
   when the source reports GNSS or heights.
5. Treat every conversion/transformation as provenance: retain source and target
   references, operation type, reported factor/grid/model, and result; do not
   backfill an unreported transformation or grid factor.
6. Store quality statements as reported, including whether calculated before
   adjustment. A computation may test consistency, but it cannot substitute a
   surveyor's reported class, certificate, or legal conclusion.

## Sources

- **[S1] North Carolina General Assembly**, [G.S. 47-30, *Plats and
  subdivisions; mapping requirements*](https://www.ncleg.gov/EnactedLegislation/Statutes/HTML/BySection/Chapter_47/GS_47-30.html).
- **[S2] North Carolina Board of Examiners for Engineers and Surveyors**,
  [21 NCAC 56, Section .1600, *Standards of Practice for Land Surveying in
  North Carolina*](https://www.ncbels.org/wp-content/uploads/2023/03/21-NCAC-56-Board-Rules.pdf)
  (Board-published administrative-rule compilation; relevant rules list amendments through July 1, 2020).
- **[S3] North Carolina General Assembly**, [G.S. 102-1.1, *Name and
  description in relation to 1983 North American
  Datum*](https://www.ncleg.gov/EnactedLegislation/Statutes/HTML/BySection/Chapter_102/GS_102-1.1.html).
- **[S4] North Carolina General Assembly**, [G.S. 102-1.2, *Name and
  description in relation to the North American Terrestrial Reference Frame of
  2022*](https://www.ncleg.gov/EnactedLegislation/Statutes/HTML/BySection/Chapter_102/GS_102-1.2.html).
- **[S5] North Carolina General Assembly**, [G.S. 102-7, *Use not
  compulsory*](https://www.ncleg.gov/EnactedLegislation/Statutes/HTML/BySection/Chapter_102/GS_102-7.html).
- **[S6] Federal Geographic Data Committee**, [*Cadastral Data Content Standard
  for the National Spatial Data Infrastructure*, FGDC-STD-003-2008, version
  1.4](https://www.fgdc.gov/standards/projects/cadastral/cadastral-data-standard-v1-4.pdf).
- **[S7] NOAA National Geodetic Survey**, [*State Plane Coordinate System
  (SPCS)*](https://geodesy.noaa.gov/SPCS/).
- **[S8] NOAA National Geodetic Survey**, [*NGS Coordinate Conversion and
  Transformation Tool (NCAT)*](https://geodesy.noaa.gov/NCAT/).
- **[S9] National Institute of Standards and Technology**, [*U.S. Survey
  Foot*](https://www.nist.gov/pml/us-surveyfoot).
