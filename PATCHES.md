# libevchain integration patches

The bundled `src/libevchain` is based on the project-provided COMP3988 evidence-chain library. The following small fixes are included so the CS55 demo runs reliably:

1. `EvidenceChain.add_artefact`: index the existing artefact dictionary by `artefact_hash` when checking duplicates.
2. `EvidenceChain.bind_file`: open media files in binary mode (`rb`) instead of text mode.
3. `RelationshipType.attributes`: correct the base static method signature so it can be called at class level.
4. `artefact_audio_type.py`: correct the `audio_creation_methods` variable reference.
5. `attribute.py`: correct `DictOf` value typechecking.
6. `EvidencePass.evaluate`: correct the abstract static method signature.

These are compatibility/bug fixes only; the CS55 matching and scoring logic lives in `src/cs55_demo/`.
