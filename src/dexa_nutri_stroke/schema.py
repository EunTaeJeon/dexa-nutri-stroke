from __future__ import annotations

from .data import Schema

__all__ = [
    "CLINICAL",
    "BODY_COMPOSITION",
    "NUTRITION",
    "CATEGORICAL",
    "STRATIFY_BY",
    "default_schema",
]


CLINICAL: tuple[str, ...] = (
    "age",
    "male",
    "bmi",
    "hypertension",
    "diabetes",
    "atrial_fibrillation",
    "coronary_artery_disease",
    "congestive_heart_failure",
    "cancer",
    "dialysis",
    "smoking",
    "pre_admission_oha",
    "pre_admission_mrs",
    "initial_nihss",
    "toast_subtype",
    "onset_to_admission_hours",
    "in_hospital_infection",
    "glucose",
    "hba1c",
    "crp",
    "bun",
    "egfr",
    "d_dimer",
    "inr",
    "total_bilirubin",
    "alkaline_phosphatase",
    "free_fatty_acid",
    "homocysteine",
)

BODY_COMPOSITION: tuple[str, ...] = (
    "asm_index",
    "fat_free_mass_index",
    "fat_mass_index",
)

NUTRITION: tuple[str, ...] = (
    "pni_initial",
    "pni_followup",
)

CATEGORICAL: tuple[str, ...] = (
    "male",
    "hypertension",
    "diabetes",
    "atrial_fibrillation",
    "coronary_artery_disease",
    "congestive_heart_failure",
    "cancer",
    "dialysis",
    "smoking",
    "pre_admission_oha",
    "in_hospital_infection",
    "toast_subtype",
    "pre_admission_mrs",
)

STRATIFY_BY: tuple[str, ...] = ("male", "diabetes")


def default_schema(outcome: str = "unfavourable_outcome_3m") -> Schema:
    return Schema(
        outcome=outcome,
        clinical=CLINICAL,
        body_composition=BODY_COMPOSITION,
        nutrition=NUTRITION,
        categorical=CATEGORICAL,
        stratify_by=STRATIFY_BY,
    )
