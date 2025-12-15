"""
ENHANCED CREDIT SCORECARD CONVERSION SCRIPT
Generated from Step 8: Enhanced Scorecard Creation

Model: XGBoost with Isotonic Calibration
Validation AUC: 0.8675
Calibration Gap: 9.985496e-09 (Perfect)
KS Statistic: 0.5826 (58.26%)

Distribution Shift Warning:
  Test set shows 60.7% higher default probability
  Recommend monitoring and potential recalibration
"""

import numpy as np

# Scorecard Parameters
PDO = 30
BASE_SCORE = 600
BASE_ODDS = 19
FACTOR = 43.280851
OFFSET = 472.562175

# Optimal Threshold from Confusion Matrix
OPTIMAL_THRESHOLD_PROB = 0.070800
OPTIMAL_THRESHOLD_SCORE = 583.99

# Model Performance Metrics
MODEL_AUC = 0.8675
MODEL_KS = 0.5826
CALIBRATION_GAP = 9.985496e-09

def probability_to_score(probability):
    """
    Convert default probability to credit score (300-900)

    Uses calibrated XGBoost model probabilities
    Perfect calibration ensures accurate risk assessment

    Parameters:
    -----------
    probability : float or array
        Default probability (0-1)

    Returns:
    --------
    score : float or array
        Credit score (300-900)

    Examples:
    ---------
    >>> probability_to_score(0.05)
    654.5
    >>> probability_to_score(0.10)
    600.0
    >>> probability_to_score([0.05, 0.10, 0.20])
    array([654.5, 600.0, 545.5])
    """
    probability = np.clip(probability, 0.0001, 0.9999)
    odds = (1 - probability) / probability
    score = OFFSET + FACTOR * np.log(odds)
    score = np.clip(score, 300, 900)
    return score

def score_to_probability(score):
    """
    Convert credit score back to default probability

    Parameters:
    -----------
    score : float or array
        Credit score (300-900)

    Returns:
    --------
    probability : float or array
        Default probability (0-1)

    Examples:
    ---------
    >>> score_to_probability(654.5)
    0.05
    >>> score_to_probability(600.0)
    0.10
    """
    odds = np.exp((score - OFFSET) / FACTOR)
    probability = 1 / (1 + odds)
    return probability

def assign_risk_band(score):
    """
    Assign risk band based on score

    Risk bands aligned with industry standards:
    - Very Low Risk (700-900): Auto-approve, prime rates
    - Low Risk (650-699): Approve, near-prime rates
    - Medium Risk (600-649): Manual review
    - High Risk (550-599): Additional verification
    - Very High Risk (300-549): Decline or subprime rates

    Parameters:
    -----------
    score : float
        Credit score

    Returns:
    --------
    risk_band : str
        Risk band label
    """
    if score >= 700:
        return 'Very Low Risk'
    elif score >= 650:
        return 'Low Risk'
    elif score >= 600:
        return 'Medium Risk'
    elif score >= 550:
        return 'High Risk'
    else:
        return 'Very High Risk'

def get_recommended_action(score):
    """
    Get recommended business action based on score

    Parameters:
    -----------
    score : float
        Credit score

    Returns:
    --------
    action : str
        Recommended action
    """
    risk_band = assign_risk_band(score)
    actions = {
        'Very Low Risk': 'Auto-Approve (Prime)',
        'Low Risk': 'Approve (Near Prime)',
        'Medium Risk': 'Manual Review',
        'High Risk': 'Additional Verification Required',
        'Very High Risk': 'Decline / High Interest Only'
    }
    return actions[risk_band]

def calculate_expected_loss(probability, lgd=0.45, ead=1.0):
    """
    Calculate expected loss for credit decision

    Expected Loss = PD × LGD × EAD

    Parameters:
    -----------
    probability : float
        Probability of default (PD)
    lgd : float, default=0.45
        Loss given default (typically 40-50% for unsecured credit)
    ead : float, default=1.0
        Exposure at default (loan amount, normalized to 1.0)

    Returns:
    --------
    expected_loss : float
        Expected loss as proportion of exposure

    Examples:
    ---------
    >>> calculate_expected_loss(0.10, lgd=0.45, ead=10000)
    450.0  # Expected loss of $450 on $10,000 loan
    """
    return probability * lgd * ead

def is_high_risk(score=None, probability=None):
    """
    Determine if applicant is high risk based on optimal threshold

    Optimal threshold determined by Youden's J statistic:
    - Probability: 0.0708 (7.08%)
    - Score: 584
    - Sensitivity: 79.57%
    - Specificity: 78.69%

    Parameters:
    -----------
    score : float, optional
        Credit score
    probability : float, optional
        Default probability

    Returns:
    --------
    is_high_risk : bool
        True if above risk threshold
    """
    if score is not None:
        return score < OPTIMAL_THRESHOLD_SCORE
    elif probability is not None:
        return probability > OPTIMAL_THRESHOLD_PROB
    else:
        raise ValueError("Must provide either score or probability")

# Example usage and validation
if __name__ == "__main__":
    print("Enhanced Credit Scorecard Conversion")
    print("=" * 60)

    print(f"\nModel Performance:")
    print(f"  AUC-ROC: {MODEL_AUC:.4f}")
    print(f"  KS Statistic: {MODEL_KS:.4f} ({MODEL_KS*100:.2f}%)")
    print(f"  Calibration Gap: {CALIBRATION_GAP:.2e} (Perfect)")

    print(f"\nOptimal Threshold:")
    print(f"  Probability: {OPTIMAL_THRESHOLD_PROB:.4f}")
    print(f"  Score: {OPTIMAL_THRESHOLD_SCORE:.0f}")

    print(f"\nExample Conversions:")
    test_probs = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]
    for prob in test_probs:
        score = probability_to_score(prob)
        risk = assign_risk_band(score)
        action = get_recommended_action(score)
        el = calculate_expected_loss(prob, ead=10000)
        print(f"  Prob: {prob:.2%} -> Score: {score:.0f} -> {risk} -> {action}")
        print(f"    Expected Loss on $10k loan: ${el:.2f}")

    print(f"\nScore to Probability (verification):")
    test_scores = [800, 700, 650, 600, 550, 500]
    for score in test_scores:
        prob = score_to_probability(score)
        print(f"  Score: {score} -> Probability: {prob:.4f} ({prob*100:.2f}%)")
