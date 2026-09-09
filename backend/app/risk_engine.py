# ============================================================
# NexaShield AI - Risk Engine
# ============================================================


def calculate_risk(
    average_synthetic_probability,
    maximum_synthetic_probability,
    suspicious_windows,
    total_windows
):

    # Convert probability to percentage
    average_score = average_synthetic_probability * 100
    maximum_score = maximum_synthetic_probability * 100


    # --------------------------------------------------------
    # Suspicious-window ratio
    # --------------------------------------------------------

    if total_windows > 0:

        suspicious_ratio = (
            suspicious_windows / total_windows
        )

    else:

        suspicious_ratio = 0


    # --------------------------------------------------------
    # Base risk
    # --------------------------------------------------------

    risk_score = average_score


    # --------------------------------------------------------
    # Evidence from strongest suspicious window
    # --------------------------------------------------------

    if maximum_score >= 90:

        risk_score += 10

    elif maximum_score >= 70:

        risk_score += 5


    # --------------------------------------------------------
    # Evidence from multiple suspicious windows
    # --------------------------------------------------------

    if suspicious_ratio >= 0.75:

        risk_score += 15

    elif suspicious_ratio >= 0.50:

        risk_score += 10

    elif suspicious_ratio >= 0.25:

        risk_score += 5


    # --------------------------------------------------------
    # Keep score between 0 and 100
    # --------------------------------------------------------

    risk_score = min(
        100,
        max(0, risk_score)
    )


    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if risk_score < 30:

        risk_level = "LOW"
        decision = "ALLOW"
        recommendation = "Voice appears genuine."


    elif risk_score < 60:

        risk_level = "MEDIUM"
        decision = "VERIFY"
        recommendation = (
            "Voice authenticity is uncertain. "
            "Additional verification recommended."
        )


    elif risk_score < 80:

        risk_level = "HIGH"
        decision = "VERIFY"
        recommendation = (
            "Suspicious voice characteristics detected. "
            "Require strong identity verification."
        )


    else:

        risk_level = "CRITICAL"
        decision = "BLOCK"
        recommendation = (
            "High probability of voice impersonation. "
            "Block or hold the transaction."
        )


    return {
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "decision": decision,
        "recommendation": recommendation,
        "average_synthetic_probability": round(
            average_score,
            2
        ),
        "maximum_synthetic_probability": round(
            maximum_score,
            2
        ),
        "suspicious_windows": suspicious_windows,
        "total_windows": total_windows
    }


# ============================================================
# Quick test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NexaShield AI - Risk Engine Test")
    print("=" * 60)


    test_cases = [

        {
            "name": "Clearly Genuine",
            "avg": 0.05,
            "max": 0.10,
            "suspicious": 0,
            "windows": 3
        },

        {
            "name": "Uncertain",
            "avg": 0.45,
            "max": 0.80,
            "suspicious": 1,
            "windows": 3
        },

        {
            "name": "Highly Suspicious",
            "avg": 0.80,
            "max": 0.99,
            "suspicious": 3,
            "windows": 3
        }

    ]


    for case in test_cases:

        result = calculate_risk(
            case["avg"],
            case["max"],
            case["suspicious"],
            case["windows"]
        )


        print()
        print(case["name"])
        print("-" * 60)

        print(
            "Risk Score:",
            result["risk_score"]
        )

        print(
            "Risk Level:",
            result["risk_level"]
        )

        print(
            "Decision:",
            result["decision"]
        )

        print(
            "Recommendation:",
            result["recommendation"]
        )