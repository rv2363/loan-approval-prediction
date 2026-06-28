from flask import Flask, render_template, request, jsonify
import pickle
import os
import time

app = Flask(__name__)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

model = pickle.load(open(os.path.join(MODEL_DIR, "loan_model.pkl"), "rb"))
encoders = pickle.load(open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb"))


@app.route("/")
def home():
    return render_template("index.html")


# ---------------- CHATBOT ----------------
@app.route("/chatbot", methods=["POST"])
def chatbot():

    user_msg = request.form["message"].lower()

    if "cibil" in user_msg:
        reply = "Banks usually prefer CIBIL score above 750 for easier loan approval."

    elif "loan rejected" in user_msg or "rejected" in user_msg:
        reply = "Possible reasons: Low CIBIL score, previous default, low income or high loan amount."

    elif "documents" in user_msg:
        reply = "Required documents: Aadhaar Card, PAN Card, Income Proof and Bank Statement."

    elif "improve" in user_msg:
        reply = "Improve CIBIL by paying EMIs on time and avoiding late payments."

    elif "age" in user_msg:
        reply = "Applicants below 18 years are not legally eligible for loan."

    elif "hello" in user_msg or "hi" in user_msg:
        reply = "Hello! I am your Smart AI Loan Assistant."

    else:
        reply = "Ask me about CIBIL score, loan rejection, approval or banking."

    return jsonify({"reply": reply})


# ---------------- PREDICTION ----------------
@app.route("/predict", methods=["POST"])
def predict():

    try:

        age = int(request.form["person_age"])
        income = float(request.form["person_income"])
        occupation = request.form["occupation"]
        education = request.form["education"]
        purpose = request.form["loan_purpose"]
        amount = float(request.form["loan_amnt"])
        cibil = int(request.form["cibil_score"])
        default = request.form["cb_person_default_on_file"]
        credit_hist = float(request.form["cb_person_cred_hist_length"])

        # Rule 1 Age check
        if age < 18:
            return render_template(
                "index.html",
                prediction_text="Loan Rejected ❌",
                confidence=100,
                ai_text="Applicant is below legal minimum age.",
                reasons=["Applicant age below 18 years"],
                cibil=cibil,
                risk="HIGH"
            )

        # Occupation mapping
        if occupation == "Salaried":
            home = "MORTGAGE"
            emp_length = 5

        elif occupation == "Business Owner":
            home = "OWN"
            emp_length = 7

        elif occupation == "Freelancer":
            home = "RENT"
            emp_length = 2

        else:
            home = "RENT"
            emp_length = 0

        # Education mapping
        if education == "Post Graduate":
            grade = "A"

        elif education == "Graduate":
            grade = "B"

        elif education == "12th Pass":
            grade = "C"

        else:
            grade = "D"

        int_rate = 9.5
        loan_percent_income = amount / income

        # Encoding
        home = encoders["person_home_ownership"].transform([home])[0]
        intent = encoders["loan_intent"].transform([purpose])[0]
        grade = encoders["loan_grade"].transform([grade])[0]
        default_encoded = encoders["cb_person_default_on_file"].transform([default])[0]

        # Model input
        data = [[
            age,
            income,
            home,
            emp_length,
            intent,
            grade,
            amount,
            int_rate,
            loan_percent_income,
            default_encoded,
            credit_hist
        ]]

        # Fake AI delay
        time.sleep(4)

        prediction = model.predict(data)[0]
        confidence = round(max(model.predict_proba(data)[0]) * 100, 2)

        reasons = []

        # AI reasons
        if cibil < 650:
            reasons.append("Low CIBIL score detected")

        if cibil >= 750:
            reasons.append("Strong CIBIL score improves approval chances")

        if loan_percent_income > 0.50:
            reasons.append("Loan amount high compared to income")

        if credit_hist < 3:
            reasons.append("Low credit history")

        if default == "Y":
            reasons.append("Previous loan default found")

        if income < 20000:
            reasons.append("Income level is low")

        # Hard rejection rules
        if cibil < 500:
            prediction = 0
            reasons.append("Extremely poor CIBIL score")

        if amount > income * 3:
            prediction = 0
            reasons.append("Requested amount exceeds repayment capacity")

        # Risk level
        if cibil >= 800:
            risk = "VERY LOW"

        elif cibil >= 700:
            risk = "LOW"

        elif cibil >= 600:
            risk = "MEDIUM"

        else:
            risk = "HIGH"

        # Final decision
        if prediction == 1 and cibil >= 650 and default != "Y":

            result = "Loan Approved ✅"
            ai_text = "AI detected strong repayment capability and healthy credit profile."

        else:

            result = "Loan Rejected ❌"
            ai_text = "AI detected multiple financial risk indicators."

        if len(reasons) == 0:
            reasons.append("Financial profile looks stable")

        return render_template(
            "index.html",
            prediction_text=result,
            confidence=confidence,
            ai_text=ai_text,
            reasons=reasons,
            cibil=cibil,
            risk=risk
        )

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)