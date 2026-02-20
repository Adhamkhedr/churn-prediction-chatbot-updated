# Chatbot User Guide -For the Marketing Team

## What Is This?

This is a chatbot that helps you check if a customer is at risk of leaving (churning). Instead of filling out forms or spreadsheets, you simply describe the customer in your own words and the chatbot will tell you:

- How likely they are to churn (a percentage)
- Whether the risk is LOW, MODERATE, or HIGH
- The top 3 reasons driving that specific customer's risk
- A recommendation on what to do next

---

## How to Use It

### Step 1: Describe the customer

Just type what you know about the customer. You can say it however you like:

> "I have a customer who's been with us for 3 months on a month-to-month contract, paying $85 a month with fiber optic internet. They don't have tech support or online security."

### Step 2: Answer follow-up questions (if any)

If you didn't mention everything the chatbot needs, it will ask you for the missing details. Just answer naturally.

### Step 3: Get the prediction

Once the chatbot has enough information, it gives you the result immediately.

### Checking another customer

Click the **"New Session"** button at the top to start fresh with a different customer.

---

## What Information Does It Need?

The chatbot needs at minimum these 6 details about the customer:

| Detail | What to say | Examples |
|--------|-------------|---------|
| How long they've been a customer | Tenure in months or years | "3 months", "2 years" |
| Contract type | What contract they're on | "month-to-month", "one year", "two year" |
| Monthly bill | How much they pay per month | "$70", "85 dollars a month" |
| Internet type | What internet service they have | "fiber optic", "DSL", "no internet" |
| Tech support | Whether they have tech support | "has tech support", "no tech support" |
| Online security | Whether they have online security | "has online security", "no online security" |

If the prediction is uncertain, the chatbot may ask for a few more details like marital status, payment method, and streaming services.

---

## Understanding the Results

### Risk Levels

| Probability | Risk Level | What It Means |
|-------------|-----------|---------------|
| Under 20% | LOW | This customer is likely to stay. No immediate action needed. |
| 20 - 50% | MODERATE | Worth keeping an eye on. Consider a check-in. |
| 50 - 80% | MODERATE-HIGH | At risk. Recommend proactive retention outreach. |
| Over 80% | HIGH | Very likely to leave. Prioritize retention efforts. |

### Risk Factors

The chatbot shows the top 3 factors that matter most for **this specific customer**. Each factor tells you whether it is pushing the customer **toward leaving** or **toward staying**. For example:

- "Tenure (Months) -increasing churn risk" means their short tenure is making them more likely to leave
- "Two Year Contract -reducing churn risk" means their contract is keeping them loyal

These factors change from customer to customer -they are not a fixed list.

---

## Suggested Actions

Based on the risk level, here are some things you can do:

**For HIGH risk customers:**
- Offer a contract upgrade (e.g., move from month-to-month to a one or two year plan) with a discount
- Bundle add-ons like tech support or online security at a reduced rate
- Have the retention team reach out directly
- Offer loyalty rewards or credits

**For MODERATE risk customers:**
- Schedule a periodic check-in
- Send targeted offers for services they don't have yet
- Monitor their usage and billing patterns

**For LOW risk customers:**
- No immediate action needed
- Continue standard engagement

---

## Example Conversations

### All info in one message
**You:** "Customer with 5 months tenure, month-to-month, $90/month, fiber optic, no tech support, no online security"

**Chatbot:** HIGH risk, ~85% probability. Top factors: Tenure (increasing risk), Online Security (increasing risk), Fiber Optic Internet (increasing risk). Recommendation: Proactive retention measures advised.

### Providing info gradually
**You:** "I want to check a customer on a two year contract with DSL"

**Chatbot:** "Got it! Could you also tell me how long this customer has been with us, their monthly charges, and whether they have tech support and online security?"

**You:** "They've been here 48 months, pay $50, have tech support and online security"

**Chatbot:** LOW risk, ~12% probability. Top factors: Two Year Contract (reducing risk), Tenure (reducing risk), Online Security (reducing risk). Recommendation: Customer appears stable.

### Only partial info
**You:** "The customer pays $70 a month"

**Chatbot:** Asks for the remaining details (tenure, contract, internet type, tech support, online security).
