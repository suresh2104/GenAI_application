_SEVERITY_MAP = {"low": 1, "medium": 2, "moderate": 2, "high": 3, "severe": 4, "critical": 5}

def _severity_score(value):
    if isinstance(value, (int, float)):
        return value
    return _SEVERITY_MAP.get(str(value).lower().strip(), 0)

def process_results(analysis_results):
    if not analysis_results:
        return {}
    final_result = max(
        analysis_results,
        key=lambda x: _severity_score(x.get('severity', 0)) if isinstance(x, dict) else 0
    )
    decision = final_result.get('decision', 'investigate').lower()
    if decision == 'approve':
        final_result['justification'] = "Damage covered under policy"
    elif decision == 'deny':
        final_result['justification'] = "Damage not covered by policy terms"
    else:
        final_result['justification'] = "Requires human investigation"
    return final_result