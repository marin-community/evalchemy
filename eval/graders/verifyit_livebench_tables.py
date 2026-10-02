"""Parse LiveBench tables for core structural and numeric comparison."""

import math

from verifyit.grade import Aggregation, InvalidTask, aggregate_rewards
from verifyit.modes.grade_json_schema import grade_json_schema_candidate
from verifyit.modes.grade_math import grade_numeric_candidate
from verifyit.spec import NumericSpec


def _records(table):
    import pandas as pd

    columns = [str(column).strip() for column in table.columns]
    if len(set(columns)) != len(columns):
        raise ValueError("Duplicate normalized table columns")
    rows = []
    for values in table.itertuples(index=False, name=None):
        row = {}
        for column, value in zip(columns, values, strict=True):
            if pd.isna(value):
                value = None
            elif isinstance(value, str):
                value = value.strip()
            elif hasattr(value, "item"):
                value = value.item()
            row[column] = value
        rows.append(row)
    return rows


def grade_table(question, response):
    from livebench.process_results.data_analysis.tablereformat import utils

    try:
        output_format = (
            question["turns"][0]
            .split("Please convert the Input Table from ")[1]
            .split("format to ")[1]
            .split(" format")[0]
            .lower()
        )
        table = utils.read_df_func(output_format, question["ground_truth"])
        if table is None or table.empty:
            raise ValueError("Empty trusted table")
        reference = _records(table)
    except (ValueError, TypeError, IndexError, AttributeError) as error:
        raise InvalidTask("Invalid trusted table format or reference") from error
    numeric = []
    rows = []
    for index, row in enumerate(reference):
        properties = {}
        for column, value in row.items():
            if type(value) in (int, float):
                spec = NumericSpec(expected=value, tolerance_abs=math.nextafter(1e-6, 0.0), tolerance_rel=0.0)
                grade_numeric_candidate(spec, value)
                numeric.append((index, column, spec))
                properties[column] = {"type": "number"}
            else:
                properties[column] = {"const": value}
        rows.append({"type": "object", "required": list(row), "additionalProperties": False, "properties": properties})
    schema = {"type": "array", "minItems": len(rows), "maxItems": len(rows), "prefixItems": rows}
    grade_json_schema_candidate(schema, reference)
    clean = utils.remove_initial_phrase(utils.clean_llm_output(response))
    candidates = []
    initial_parse_failed = False
    try:
        candidates.append(utils.read_df_func(output_format, clean))
    except (ValueError, TypeError, IndexError):
        initial_parse_failed = True
    # These source helpers only parse text; acceptance and retry selection belong to core.
    if output_format == "csv" or (output_format == "tsv" and initial_parse_failed):
        separator = "," if output_format == "csv" else "\t"
        candidates.append(utils.read_sep_table_from_text(clean, separator.join(table.columns), sep=separator))
    elif output_format == "jsonl":
        candidates.append(utils.read_jsonl_table_from_text(clean, table.columns))
    verdicts = []
    for candidate in candidates:
        try:
            records = _records(candidate) if candidate is not None else None
        except (ValueError, TypeError, AttributeError):
            records = None
        components = [grade_json_schema_candidate(schema, records)]
        for index, column, spec in numeric:
            value = records[index].get(column, math.nan) if records is not None and index < len(records) else math.nan
            components.append(grade_numeric_candidate(spec, value if type(value) in (int, float) else math.nan))
        verdicts.append(aggregate_rewards(components, expected_total=1 + len(numeric), policy=Aggregation.ALL))
    if not verdicts:
        verdicts.append(grade_json_schema_candidate(schema, None))
    return aggregate_rewards(verdicts, expected_total=len(verdicts), policy=Aggregation.MAX)
