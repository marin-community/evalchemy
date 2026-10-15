"""Paired-condition reduction over compact per-example measurements."""

from collections import defaultdict


class PairedConditions:
    def __init__(self):
        self.rows = []

    def add(self, item):
        self.rows.append((dict(item.spec.tags), item.spec.grader.reference["pair_id"], dict(item.grade.metrics)))

    def finish(self):
        baseline = {pair: scores["accuracy"] for tags, pair, scores in self.rows if tags["condition"] == "baseline"}
        tasks = defaultdict(list)
        for row in self.rows:
            tasks[row[0]["task"]].append(row)
        result = {}
        def average(rows, metric):
            return 100 * sum(scores[metric] for _, _, scores in rows) / len(rows)
        for task, rows in tasks.items():
            result[task] = average(rows, "accuracy")
            result[task + "_no_answer"] = average(rows, "no_answer")
            conditions = defaultdict(list)
            for row in rows:
                if row[0]["condition"] != "baseline":
                    conditions[row[0]["condition"]].append(row)
            for condition, members in conditions.items():
                prefix = f"{task}:{condition}"
                result[prefix] = average(members, "accuracy")
                result[prefix + "_no_answer"] = average(members, "no_answer")
                paired = [row for row in members if row[1] in baseline]
                if paired:
                    clean = 100 * sum(baseline[pair] for _, pair, _ in paired) / len(paired)
                    result[prefix + "_clean_subset"] = clean
                    result[prefix + "_paired_delta"] = result[prefix] - clean
                    result[prefix + "_harmed"] = sum(baseline[pair] and not scores["accuracy"] for _, pair, scores in paired)
                    result[prefix + "_helped"] = sum(scores["accuracy"] and not baseline[pair] for _, pair, scores in paired)
                changed = [row for row in members if row[0]["changed"] == "True"]
                if changed and len(changed) != len(members):
                    result[prefix + "_changed_only"] = average(changed, "accuracy")
                number_words = [row for row in members if row[0]["number_words"] == "True"]
                if number_words:
                    result[prefix + "_number_words"] = average(number_words, "accuracy")
        return result
