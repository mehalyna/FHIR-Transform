"""
Frozen R0 discovery/loading and R0 -> R1 -> R2 transformation logic, extracted verbatim from
notebooks/01_fhir_dataset_audit.ipynb and notebooks/03_representation_transformations.ipynb so
that Stage 5 can benchmark the *actual* transformation cost without re-deriving or altering it.

This module is a refactoring-for-reuse only: every function below reproduces the exact behavior
of the corresponding notebook cell (including the Stage 3 methodological correction to
MedicationRequest.reasonReference -> Condition in R2). Nothing here changes what R0, R1 or R2
contain. Correctness is verified in the Stage 5 notebook (Section 3) by comparing the DataFrames
built here against the frozen results/stage3/*.csv files byte-for-byte.
"""
import itertools
import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd

EXCLUDE_DIRS = {".git", ".venv", "venv", "node_modules", "notebooks", "results", ".ipynb_checkpoints", "__pycache__"}
MIN_JSON_FILES_FOR_CANDIDATE = 10
TARGET_PATIENT_COUNT = 100
TARGET_RESOURCE_TYPES = ["Patient", "Encounter", "Condition", "Observation", "MedicationRequest"]


def find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists():
            return parent
    return start.resolve()


def discover_dataset_dir(repo_root: Path) -> Path:
    """Same dataset-discovery walk used identically in notebooks 01/03/04."""
    candidates = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        json_files_here = [f for f in filenames if f.lower().endswith(".json")]
        if len(json_files_here) >= MIN_JSON_FILES_FOR_CANDIDATE:
            total_size = sum((Path(dirpath) / f).stat().st_size for f in json_files_here)
            candidates.append({"path": Path(dirpath), "n_json_files": len(json_files_here), "total_size_bytes": total_size})
    best = min(candidates, key=lambda c: abs(c["n_json_files"] - TARGET_PATIENT_COUNT))
    return best["path"]


def ref_id(ref_obj):
    """Extract the bare target id from a FHIR Reference object (Stage 1/3 logic, verbatim)."""
    if not ref_obj:
        return None
    ref = ref_obj.get("reference")
    if not ref:
        return None
    if ref.startswith("urn:uuid:"):
        return ref[len("urn:uuid:"):]
    m = re.match(r"^[A-Za-z]+/(.+)$", ref)
    return m.group(1) if m else ref


def load_r0(dataset_dir: Path):
    """Parse every FHIR Bundle JSON file in dataset_dir (verbatim NB03/NB04 loading logic).

    Returns (all_resources, resources_by_target_type).
    """
    json_files = sorted(Path(dataset_dir).glob("*.json"))
    raw_data = {fp.name: json.load(open(fp, encoding="utf-8")) for fp in json_files}
    all_resources = [e["resource"] for data in raw_data.values() for e in data.get("entry", []) if e.get("resource") is not None]
    resources_by_target_type = {rt: [r for r in all_resources if r.get("resourceType") == rt] for rt in TARGET_RESOURCE_TYPES}
    return all_resources, resources_by_target_type


def load_r0_filtered(dataset_dir: Path, needed_types):
    """Parse every FHIR Bundle JSON file, keeping only resources whose type is in needed_types.

    Every patient bundle interleaves all resource types, so the full JSON of every file must still
    be parsed (json.load cannot selectively deserialize part of a file); only the *filtering* step
    is workload-specific. Used for the R0 end-to-end / loading-cost benchmarks in Stage 5 so that a
    workload never pays for building resource lists it does not need, while not pretending R0 can
    skip parsing files it does not need (it cannot, without an out-of-band type index R0 does not
    have).
    """
    needed = set(needed_types)
    json_files = sorted(Path(dataset_dir).glob("*.json"))
    resources_by_type = {rt: [] for rt in needed}
    for fp in json_files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        for e in data.get("entry", []):
            res = e.get("resource")
            if res is not None and res.get("resourceType") in needed:
                resources_by_type[res.get("resourceType")].append(res)
    return resources_by_type


def first_coding(codeable_concept):
    """Deterministic R2 rule: keep coding[0] only (NB03 Section 11, verbatim)."""
    if not codeable_concept:
        return None, None, 0
    codings = codeable_concept.get("coding", []) or []
    if not codings:
        return None, None, 0
    n_discarded = len(codings) - 1
    return codings[0].get("system"), codings[0].get("code"), n_discarded


def build_r1_tables(resources_by_target_type):
    """Exact reproduction of notebooks/03_representation_transformations.ipynb Sections 2-8."""
    _concept_id_seq = itertools.count(1)
    coded_concept_rows = []
    coding_rows = []
    identifier_rows = []
    participant_rows = []
    relationship_rows = []
    dosage_rows = []
    component_rows = []

    def add_coded_concept(resource_type, resource_id, field_path, codeable_concept):
        if not codeable_concept:
            return None
        concept_id = next(_concept_id_seq)
        coded_concept_rows.append({
            "concept_id": concept_id, "resource_type": resource_type, "resource_id": resource_id,
            "field_path": field_path, "text": codeable_concept.get("text"),
        })
        for i, coding in enumerate(codeable_concept.get("coding", []) or []):
            coding_rows.append({
                "concept_id": concept_id, "coding_ordinal": i,
                "system": coding.get("system"), "code": coding.get("code"), "display": coding.get("display"),
            })
        return concept_id

    def add_bare_coding_as_concept(resource_type, resource_id, field_path, coding):
        if not coding:
            return None
        concept_id = next(_concept_id_seq)
        coded_concept_rows.append({
            "concept_id": concept_id, "resource_type": resource_type, "resource_id": resource_id,
            "field_path": field_path, "text": None,
        })
        coding_rows.append({
            "concept_id": concept_id, "coding_ordinal": 0,
            "system": coding.get("system"), "code": coding.get("code"), "display": coding.get("display"),
        })
        return concept_id

    def extract_value(resource_type, resource_id, field_prefix, obj):
        if "valueQuantity" in obj:
            vq = obj["valueQuantity"] or {}
            return {
                "value_type": "Quantity", "value_quantity_value": vq.get("value"), "value_quantity_unit": vq.get("unit"),
                "value_quantity_system": vq.get("system"), "value_quantity_code": vq.get("code"),
                "value_codeable_concept_id": None, "value_string": None,
            }
        if "valueCodeableConcept" in obj:
            concept_id = add_coded_concept(resource_type, resource_id, f"{field_prefix}.valueCodeableConcept", obj["valueCodeableConcept"])
            return {
                "value_type": "CodeableConcept", "value_quantity_value": None, "value_quantity_unit": None,
                "value_quantity_system": None, "value_quantity_code": None,
                "value_codeable_concept_id": concept_id, "value_string": None,
            }
        if "valueString" in obj:
            return {
                "value_type": "String", "value_quantity_value": None, "value_quantity_unit": None,
                "value_quantity_system": None, "value_quantity_code": None,
                "value_codeable_concept_id": None, "value_string": obj["valueString"],
            }
        return {
            "value_type": "None", "value_quantity_value": None, "value_quantity_unit": None,
            "value_quantity_system": None, "value_quantity_code": None,
            "value_codeable_concept_id": None, "value_string": None,
        }

    patient_rows = []
    for p in resources_by_target_type["Patient"]:
        pid = p["id"]
        patient_rows.append({"patient_id": pid, "birth_date": p.get("birthDate"), "deceased_datetime": p.get("deceasedDateTime")})
        for i, ident in enumerate(p.get("identifier", []) or []):
            identifier_rows.append({
                "resource_type": "Patient", "resource_id": pid, "identifier_ordinal": i,
                "system": ident.get("system"), "value": ident.get("value"), "use": ident.get("use"),
            })

    encounter_rows = []
    for e in resources_by_target_type["Encounter"]:
        eid = e["id"]
        patient_id = ref_id(e.get("subject"))
        class_concept_id = add_bare_coding_as_concept("Encounter", eid, "class", e.get("class"))
        for i, ident in enumerate(e.get("identifier", []) or []):
            identifier_rows.append({
                "resource_type": "Encounter", "resource_id": eid, "identifier_ordinal": i,
                "system": ident.get("system"), "value": ident.get("value"), "use": ident.get("use"),
            })
        for i, tcc in enumerate(e.get("type", []) or []):
            add_coded_concept("Encounter", eid, f"type[{i}]", tcc)
        for i, part in enumerate(e.get("participant", []) or []):
            individual = part.get("individual") or {}
            ref = individual.get("reference")
            target_type_hint = None
            if ref:
                m = re.match(r"^([A-Za-z]+)[?/]", ref)
                if m:
                    target_type_hint = m.group(1)
            period = part.get("period") or {}
            participant_rows.append({
                "encounter_id": eid, "participant_ordinal": i,
                "individual_resource_type": target_type_hint, "individual_reference": ref,
                "period_start": period.get("start"), "period_end": period.get("end"),
            })
        period = e.get("period") or {}
        encounter_rows.append({
            "encounter_id": eid, "patient_id": patient_id, "period_start": period.get("start"),
            "period_end": period.get("end"), "class_concept_id": class_concept_id,
        })
        if patient_id is not None:
            relationship_rows.append({"source_resourceType": "Encounter", "source_id": eid, "reference_field": "subject",
                                       "target_resourceType": "Patient", "target_id": patient_id})

    condition_rows = []
    for c in resources_by_target_type["Condition"]:
        cid = c["id"]
        patient_id = ref_id(c.get("subject"))
        encounter_id = ref_id(c.get("encounter"))
        code_concept_id = add_coded_concept("Condition", cid, "code", c.get("code"))
        cs_concept_id = add_coded_concept("Condition", cid, "clinicalStatus", c.get("clinicalStatus"))
        vs_concept_id = add_coded_concept("Condition", cid, "verificationStatus", c.get("verificationStatus"))
        condition_rows.append({
            "condition_id": cid, "patient_id": patient_id, "encounter_id": encounter_id,
            "code_concept_id": code_concept_id, "clinical_status_concept_id": cs_concept_id,
            "verification_status_concept_id": vs_concept_id, "onset_datetime": c.get("onsetDateTime"),
            "recorded_date": c.get("recordedDate"), "abatement_datetime": c.get("abatementDateTime"),
        })
        if patient_id is not None:
            relationship_rows.append({"source_resourceType": "Condition", "source_id": cid, "reference_field": "subject",
                                       "target_resourceType": "Patient", "target_id": patient_id})
        if encounter_id is not None:
            relationship_rows.append({"source_resourceType": "Condition", "source_id": cid, "reference_field": "encounter",
                                       "target_resourceType": "Encounter", "target_id": encounter_id})

    observation_rows = []
    for o in resources_by_target_type["Observation"]:
        oid = o["id"]
        patient_id = ref_id(o.get("subject"))
        encounter_id = ref_id(o.get("encounter"))
        code_concept_id = add_coded_concept("Observation", oid, "code", o.get("code"))
        for i, cat in enumerate(o.get("category", []) or []):
            add_coded_concept("Observation", oid, f"category[{i}]", cat)
        value_fields = extract_value("Observation", oid, "", o)
        components = o.get("component", []) or []
        for i, comp in enumerate(components):
            comp_code_concept_id = add_coded_concept("Observation", oid, f"component[{i}].code", comp.get("code"))
            comp_value_fields = extract_value("Observation", oid, f"component[{i}]", comp)
            component_rows.append({"observation_id": oid, "component_ordinal": i, "code_concept_id": comp_code_concept_id, **comp_value_fields})
        observation_rows.append({
            "observation_id": oid, "patient_id": patient_id, "encounter_id": encounter_id,
            "effective_datetime": o.get("effectiveDateTime"), "issued": o.get("issued"),
            "code_concept_id": code_concept_id, "has_component": len(components) > 0, **value_fields,
        })
        if patient_id is not None:
            relationship_rows.append({"source_resourceType": "Observation", "source_id": oid, "reference_field": "subject",
                                       "target_resourceType": "Patient", "target_id": patient_id})
        if encounter_id is not None:
            relationship_rows.append({"source_resourceType": "Observation", "source_id": oid, "reference_field": "encounter",
                                       "target_resourceType": "Encounter", "target_id": encounter_id})

    medication_request_rows = []
    for m in resources_by_target_type["MedicationRequest"]:
        mid = m["id"]
        patient_id = ref_id(m.get("subject"))
        encounter_id = ref_id(m.get("encounter"))
        med_cc_concept_id = add_coded_concept("MedicationRequest", mid, "medicationCodeableConcept", m.get("medicationCodeableConcept")) if "medicationCodeableConcept" in m else None
        med_ref_id = ref_id(m.get("medicationReference")) if "medicationReference" in m else None
        reason_refs = m.get("reasonReference", []) or []
        reason_condition_id = ref_id(reason_refs[0]) if reason_refs else None
        for i, di in enumerate(m.get("dosageInstruction", []) or []):
            dosage_rows.append({"medication_request_id": mid, "dosage_ordinal": i, "text": di.get("text"), "as_needed_boolean": di.get("asNeededBoolean")})
        medication_request_rows.append({
            "medication_request_id": mid, "patient_id": patient_id, "encounter_id": encounter_id,
            "medication_codeable_concept_id": med_cc_concept_id, "medication_reference_id": med_ref_id,
            "authored_on": m.get("authoredOn"), "reason_condition_id": reason_condition_id,
        })
        if patient_id is not None:
            relationship_rows.append({"source_resourceType": "MedicationRequest", "source_id": mid, "reference_field": "subject",
                                       "target_resourceType": "Patient", "target_id": patient_id})
        if encounter_id is not None:
            relationship_rows.append({"source_resourceType": "MedicationRequest", "source_id": mid, "reference_field": "encounter",
                                       "target_resourceType": "Encounter", "target_id": encounter_id})
        if reason_condition_id is not None:
            relationship_rows.append({"source_resourceType": "MedicationRequest", "source_id": mid, "reference_field": "reasonReference",
                                       "target_resourceType": "Condition", "target_id": reason_condition_id})

    return {
        "r1_patient": pd.DataFrame(patient_rows),
        "r1_encounter": pd.DataFrame(encounter_rows),
        "r1_encounter_participant": pd.DataFrame(participant_rows),
        "r1_condition": pd.DataFrame(condition_rows),
        "r1_observation": pd.DataFrame(observation_rows),
        "r1_observation_component": pd.DataFrame(component_rows),
        "r1_medication_request": pd.DataFrame(medication_request_rows),
        "r1_dosage": pd.DataFrame(dosage_rows),
        "r1_coded_concept": pd.DataFrame(coded_concept_rows),
        "r1_coding": pd.DataFrame(coding_rows),
        "r1_relationship": pd.DataFrame(relationship_rows),
        "r1_identifier": pd.DataFrame(identifier_rows),
    }


def build_r2_tables(resources_by_target_type, all_resources, r1_tables):
    """Exact reproduction of NB03 Sections 11-15, including the Stage 3 methodological correction
    to MedicationRequest.reasonReference -> Condition.

    Depends on r1_tables["r1_patient"] and r1_tables["r1_condition"], exactly as the frozen
    notebook's R2 build does (r2_patient is a direct column slice of r1_patient_df; the corrected
    reason_condition_id is validated against r1_condition_df's condition_id set). R2 is therefore
    not independently constructible from R0 alone in the current frozen pipeline -- this is a real
    property of the Stage 3 implementation, reported as-is rather than bypassed.
    """
    r1_patient_df = r1_tables["r1_patient"]
    r1_condition_df = r1_tables["r1_condition"]

    component_meta = {}
    for o in resources_by_target_type["Observation"]:
        for comp in o.get("component", []) or []:
            for coding in (comp.get("code") or {}).get("coding", []) or []:
                code = coding.get("code")
                meta = component_meta.setdefault(code, {"system": coding.get("system"), "display": coding.get("display"), "value_types": Counter()})
                if "valueQuantity" in comp:
                    meta["value_types"]["Quantity"] += 1
                elif "valueCodeableConcept" in comp:
                    meta["value_types"]["CodeableConcept"] += 1
                elif "valueString" in comp:
                    meta["value_types"]["String"] += 1
                else:
                    meta["value_types"]["None"] += 1

    def slugify(code):
        return re.sub(r"[^0-9A-Za-z]+", "_", code)

    component_schema = {}
    for code, meta in component_meta.items():
        dominant_type, dominant_n = meta["value_types"].most_common(1)[0]
        total = sum(meta["value_types"].values())
        component_schema[code] = {
            "slug": slugify(code), "system": meta["system"], "display": meta["display"],
            "value_type": dominant_type, "n_occurrences": total, "n_mixed_type_instances": total - dominant_n,
        }
    component_schema_df = pd.DataFrame([{"code": k, **v} for k, v in component_schema.items()]).sort_values("n_occurrences", ascending=False)

    r2_observation_rows = []
    for o in resources_by_target_type["Observation"]:
        oid = o["id"]
        patient_id = ref_id(o.get("subject"))
        encounter_id = ref_id(o.get("encounter"))
        code_system, code_code, _ = first_coding(o.get("code"))
        categories = o.get("category", []) or []
        category_system, category_code, _ = first_coding(categories[0]) if categories else (None, None, 0)
        row = {
            "observation_id": oid, "patient_id": patient_id, "encounter_id": encounter_id,
            "event_timestamp": o.get("effectiveDateTime"), "code_system": code_system, "code_code": code_code,
            "category_system": category_system, "category_code": category_code,
            "value_type": None, "value_numeric": None, "value_unit": None,
            "value_coded_system": None, "value_coded_code": None, "value_string": None,
        }
        if "valueQuantity" in o:
            vq = o["valueQuantity"] or {}
            row.update(value_type="Quantity", value_numeric=vq.get("value"), value_unit=vq.get("unit"))
        elif "valueCodeableConcept" in o:
            vs, vc, _ = first_coding(o["valueCodeableConcept"])
            row.update(value_type="CodeableConcept", value_coded_system=vs, value_coded_code=vc)
        elif "valueString" in o:
            row.update(value_type="String", value_string=o["valueString"])
        else:
            row["value_type"] = "None"
        components = o.get("component", []) or []
        row["has_component"] = len(components) > 0
        seen_codes_this_obs = set()
        for comp in components:
            comp_codings = (comp.get("code") or {}).get("coding", []) or []
            if not comp_codings:
                continue
            code = comp_codings[0].get("code")
            if code in seen_codes_this_obs:
                continue
            seen_codes_this_obs.add(code)
            slug = component_schema[code]["slug"]
            vtype = component_schema[code]["value_type"]
            if vtype == "Quantity" and "valueQuantity" in comp:
                vq = comp["valueQuantity"] or {}
                row[f"comp_{slug}_value"] = vq.get("value")
                row[f"comp_{slug}_unit"] = vq.get("unit")
            elif vtype == "CodeableConcept" and "valueCodeableConcept" in comp:
                _, vc, _ = first_coding(comp["valueCodeableConcept"])
                row[f"comp_{slug}_code"] = vc
            elif vtype == "String" and "valueString" in comp:
                row[f"comp_{slug}_text"] = comp["valueString"]
        r2_observation_rows.append(row)
    r2_observation_df = pd.DataFrame(r2_observation_rows)

    r2_patient_df = r1_patient_df[["patient_id", "birth_date"]].copy()

    r2_encounter_rows = []
    for e in resources_by_target_type["Encounter"]:
        class_system, class_code, _ = first_coding({"coding": [e["class"]]} if e.get("class") else None)
        types = e.get("type", []) or []
        type_system, type_code, _ = first_coding(types[0]) if types else (None, None, 0)
        period = e.get("period") or {}
        r2_encounter_rows.append({
            "encounter_id": e["id"], "patient_id": ref_id(e.get("subject")), "event_timestamp": period.get("start"),
            "class_system": class_system, "class_code": class_code, "type_system": type_system, "type_code": type_code,
        })
    r2_encounter_df = pd.DataFrame(r2_encounter_rows)

    r2_condition_rows = []
    for c in resources_by_target_type["Condition"]:
        code_system, code_code, _ = first_coding(c.get("code"))
        r2_condition_rows.append({
            "condition_id": c["id"], "patient_id": ref_id(c.get("subject")), "encounter_id": ref_id(c.get("encounter")),
            "event_timestamp": c.get("onsetDateTime"), "code_system": code_system, "code_code": code_code,
        })
    r2_condition_df = pd.DataFrame(r2_condition_rows)

    id_to_type_full = {r.get("id"): r.get("resourceType") for r in all_resources if r.get("id") is not None}
    r1_condition_ids_for_check = set(r1_condition_df["condition_id"])

    r2_medication_request_rows = []
    for m in resources_by_target_type["MedicationRequest"]:
        med_system, med_code, _ = first_coding(m.get("medicationCodeableConcept")) if "medicationCodeableConcept" in m else (None, None, 0)
        condition_targeting_refs = [ref_obj for ref_obj in (m.get("reasonReference", []) or []) if id_to_type_full.get(ref_id(ref_obj)) == "Condition"]
        reason_condition_id = None
        if len(condition_targeting_refs) == 1:
            candidate_id = ref_id(condition_targeting_refs[0])
            if candidate_id in r1_condition_ids_for_check:
                reason_condition_id = candidate_id
        r2_medication_request_rows.append({
            "medication_request_id": m["id"], "patient_id": ref_id(m.get("subject")), "encounter_id": ref_id(m.get("encounter")),
            "event_timestamp": m.get("authoredOn"), "medication_system": med_system, "medication_code": med_code,
            "reason_condition_id": reason_condition_id,
        })
    r2_medication_request_df = pd.DataFrame(r2_medication_request_rows)

    return {
        "r2_patient": r2_patient_df,
        "r2_encounter": r2_encounter_df,
        "r2_condition": r2_condition_df,
        "r2_observation": r2_observation_df,
        "r2_medication_request": r2_medication_request_df,
        "r2_observation_component_schema": component_schema_df,
    }
