# Limitations and responsible-use statement

1. **Synthetic data.** PaySim is useful for controlled experimentation but cannot reproduce all behaviours, incentives and feedback loops found in real payment systems.
2. **Artificial labels.** Fraud patterns are injected by a simulator and may be easier for models to learn than genuine fraud.
3. **Limited context.** The dataset lacks device, channel, geography, authentication, merchant and network information.
4. **Ambiguous feature timing.** Post-transaction balances are excluded from the online model to avoid relying on unavailable information.
5. **Illustrative economics.** Review and error costs are explicit scenarios, not real savings estimates.
6. **No causal interpretation.** Feature importance and SHAP explain the fitted predictor, not the causes of fraud.
7. **No fairness conclusion.** The absence of protected attributes prevents a meaningful fairness audit; it does not prove the model is fair.
8. **Label delay.** Real fraud labels arrive late and can be biased by existing investigation policies.
9. **Feedback loops.** Blocking transactions changes future observations and requires policy-aware monitoring in a real system.
10. **Security.** The demonstration API lacks enterprise authentication, encryption architecture, rate limiting and a formal threat model.
11. **Scalability.** Local training and inference are demonstrations, not benchmarks for production transaction volumes.
12. **External validity.** Results must not be generalised to a specific institution, market or population.

## Responsible-use rule

The repository is for education and portfolio review. No artifact produced by it should be used to make decisions about real people or financial transactions.
