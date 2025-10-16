# Workshop Facilitator Guide

This guide is for workshop facilitators and instructors presenting the Databricks ML Workshop.

## 📋 Pre-Workshop Checklist

### 1 Week Before
- [ ] Confirm Databricks workspace access for all participants
- [ ] Test all notebooks on a fresh cluster
- [ ] Verify all dependencies are available
- [ ] Prepare backup materials in case of technical issues

### 1 Day Before
- [ ] Send pre-reading materials to participants
- [ ] Share workspace URL and login instructions
- [ ] Test screen sharing and presentation setup
- [ ] Prepare Q&A document

### Day Of
- [ ] Log in 15 minutes early
- [ ] Test cluster startup time
- [ ] Have notebooks ready to share
- [ ] Prepare example questions for discussion

## 🎯 Workshop Flow (3 Hours)

### Introduction (15 minutes)
- Welcome and introductions
- Workshop objectives overview
- Databricks platform overview
- Use case introduction: Customer loan decisions

**Key Points:**
- This is a hands-on workshop
- Participants should follow along
- Questions encouraged throughout
- Real-world MLOps focus

---

### Module 1: Data Generation (20 minutes)
**Notebook:** `01_data_generation.py`

**Teaching Points:**
- Delta Lake benefits for ML workflows
- Importance of data quality in ML
- Synthetic data generation techniques
- Feature design considerations

**Demo Flow:**
1. Explain the use case and business problem
2. Walk through data generation function
3. Show data quality checks
4. Display sample data
5. Explain Delta Lake storage

**Common Questions:**
- *Why synthetic data?* → Safe for training, no privacy concerns
- *Why Delta Lake?* → ACID transactions, time travel, performance
- *What if I have my own data?* → Easy to adapt the notebooks

**Time Management:** Keep to 20 minutes, don't dive too deep into code details

---

### Module 2: EDA & Feature Engineering (25 minutes)
**Notebook:** `02_eda_feature_engineering.py`

**Teaching Points:**
- Importance of EDA in ML projects
- Feature engineering impact on model performance
- Domain knowledge in feature creation
- Data visualization for insights

**Demo Flow:**
1. Show target variable distribution
2. Highlight key visualizations (credit score, income, etc.)
3. Explain correlation analysis
4. Walk through feature engineering logic
5. Emphasize business logic in features

**Interactive Elements:**
- Ask: "What features would you create for loan decisions?"
- Discuss: Which features show strongest correlation?
- Poll: Have you seen similar patterns in your data?

**Common Questions:**
- *How to choose features?* → Domain knowledge + correlation analysis
- *Too many features?* → We'll see feature importance later
- *What about feature scaling?* → Some models need it, shown in training

**Time Management:** 25 minutes - can speed up through visualizations if needed

---

### Module 3: Model Training (35 minutes)
**Notebook:** `03_model_training.py`

**Teaching Points:**
- MLflow experiment tracking benefits
- Hyperparameter tuning strategies
- Model comparison methodology
- MLOps best practices

**Demo Flow:**
1. Explain MLflow architecture
2. Show baseline model (Logistic Regression)
3. Demonstrate hyperparameter tuning
4. Show how experiments are logged
5. Navigate to MLflow UI during run

**Interactive Elements:**
- Live demo: Show MLflow UI
- Comparison: Discuss different algorithms
- Discussion: When to use which algorithm?

**Pro Tips:**
- Keep one MLflow tab open during training
- Point out automatic logging features
- Show how to compare runs in UI
- Explain parameter vs metric logging

**Common Questions:**
- *Why so many parameters?* → Finding optimal configuration
- *How long does training take?* → Depends on data size and compute
- *Can I use my own algorithms?* → Yes, MLflow supports any model
- *What about deep learning?* → Databricks supports TensorFlow/PyTorch too

**Time Management:** 35 minutes - this is a key module, don't rush

---

### Break (10 minutes)

Use this time to:
- Answer individual questions
- Help anyone who fell behind
- Check that everyone's notebooks are running

---

### Module 4: Model Evaluation (20 minutes)
**Notebook:** `04_model_evaluation.py`

**Teaching Points:**
- Comprehensive model evaluation
- Multiple metrics consideration
- Champion model selection
- Model Registry workflow
- Production deployment stages

**Demo Flow:**
1. Load and compare all models
2. Show model comparison visualizations
3. Explain champion model selection
4. Demo Model Registry
5. Show stage transitions

**Interactive Elements:**
- Discussion: What metrics matter most for loan decisions?
- Trade-offs: Precision vs Recall in this context
- Business impact: Cost of false positives vs false negatives

**Key Concepts:**
- Model Registry as central hub
- Version control for models
- Stage-based deployment
- Model lineage tracking

**Common Questions:**
- *Why not just use accuracy?* → Discuss class imbalance and business costs
- *How to pick champion model?* → Depends on business requirements
- *Can I rollback models?* → Yes, Model Registry supports this

**Time Management:** 20 minutes - focus on Model Registry features

---

### Module 5: Model Serving (20 minutes)
**Notebook:** `05_model_serving.py`

**Teaching Points:**
- Model serving patterns (batch vs real-time)
- Production inference considerations
- Model monitoring importance
- API design for ML models

**Demo Flow:**
1. Load model from registry
2. Show batch scoring
3. Demo real-time predictions
4. Explain inference API
5. Discuss monitoring metrics

**Interactive Elements:**
- Examples: Show different customer profiles
- Discussion: When to use batch vs real-time?
- Real-world: What monitoring metrics matter?

**Pro Tips:**
- Show prediction confidence scores
- Explain model versioning in production
- Discuss A/B testing concepts
- Mention model drift

**Common Questions:**
- *How fast is inference?* → Show timing metrics
- *Can I deploy to REST endpoint?* → Yes, Databricks Model Serving
- *What about scalability?* → Databricks handles auto-scaling
- *How to monitor in production?* → Use Delta tables for logging

**Time Management:** 20 minutes - keep practical and demo-focused

---

### Module 6: AutoML (25 minutes)
**Notebook:** `06_automl.py`

**Teaching Points:**
- AutoML benefits and use cases
- When to use AutoML vs manual ML
- Understanding generated notebooks
- Comparing AutoML with manual training

**Demo Flow:**
1. Explain AutoML concept
2. Run AutoML (or show pre-run results if time-constrained)
3. Review generated notebooks
4. Compare with manual models
5. Show feature importance from AutoML

**Interactive Elements:**
- Poll: Who has used AutoML before?
- Discussion: Where does AutoML fit in your workflow?
- Comparison: AutoML vs manual results

**Key Messages:**
- AutoML is a starting point, not replacement
- Great for quick baselines
- Generated code is customizable
- Complements manual ML expertise

**Common Questions:**
- *Is AutoML always better?* → No, it's about speed and baseline
- *Can I customize AutoML?* → Yes, via generated notebooks
- *What algorithms does it try?* → Most common ones for the problem type
- *Cost considerations?* → Depends on timeout and max_trials settings

**Pro Tips:**
- Run AutoML before workshop, show results to save time
- Have generated notebooks ready to show
- Discuss when you would/wouldn't use AutoML

**Time Management:** 25 minutes - AutoML run can take time, consider pre-running

---

### Wrap-up & Q&A (10 minutes)

**Summary Points:**
1. ✅ Complete ML workflow from data to deployment
2. ✅ MLflow for experiment tracking and model management
3. ✅ Model Registry for production deployment
4. ✅ AutoML for rapid experimentation
5. ✅ MLOps best practices throughout

**Key Takeaways:**
- Databricks provides end-to-end ML platform
- MLflow is industry-standard for ML tracking
- Model Registry enables production deployment
- AutoML accelerates initial development
- Everything is reproducible and auditable

**Next Steps for Participants:**
- Experiment with your own datasets
- Explore advanced features (model serving endpoints, drift detection)
- Try different algorithms and features
- Implement in your projects
- Share learnings with your team

**Final Q&A:**
- Open floor for any questions
- Share resources and documentation
- Provide contact information for follow-up

---

## 🎤 Presentation Tips

### General Guidelines
1. **Interactive:** Encourage questions throughout, not just at the end
2. **Practical:** Focus on "why" not just "how"
3. **Pacing:** Watch the clock, but don't rush through key concepts
4. **Engagement:** Use polls, discussions, and real-world examples
5. **Support:** Have TAs or helpers for hands-on troubleshooting

### Technical Setup
- **Two Monitors:** One for presenting, one for monitoring participants
- **Screen Share:** Share notebook, not entire screen
- **Font Size:** Increase notebook font for readability
- **Pre-run Cells:** Have some cells pre-run to save time
- **Backup Plan:** Have screenshots in case of technical issues

### Engagement Strategies
- **Polls:** Use quick polls to gauge understanding
- **Analogies:** Relate ML concepts to everyday experiences
- **Business Focus:** Always tie back to business value
- **War Stories:** Share real-world ML successes and failures
- **Hands-on:** Give time for participants to run code themselves

---

## 📊 Assessment Questions

Use these to check understanding:

### Knowledge Check 1 (After Module 2)
1. Why is exploratory data analysis important?
2. Name three engineered features and explain their business logic
3. What does correlation tell us about features?

### Knowledge Check 2 (After Module 4)
1. What is the purpose of MLflow?
2. How does Model Registry help in production deployment?
3. Why might we choose a model with slightly lower accuracy?

### Knowledge Check 3 (After Module 6)
1. When would you use AutoML vs manual training?
2. What are the main benefits of AutoML?
3. How can you customize AutoML results?

---

## 🔧 Troubleshooting Guide

### Issue: Cluster won't start
**Solution:** 
- Check cluster configuration
- Verify permissions
- Try different node type
- Use single-node mode for demos

### Issue: Import fails
**Solution:**
- Verify ML Runtime is selected
- Check Python version compatibility
- Install missing packages via cluster libraries

### Issue: Notebook hangs
**Solution:**
- Check cluster CPU/memory
- Reduce dataset size if needed
- Restart cluster if necessary

### Issue: MLflow experiment not visible
**Solution:**
- Check experiment name
- Verify notebook is attached to cluster
- Refresh MLflow UI

### Issue: Participant fell behind
**Solution:**
- Have TAs help individually
- Provide pre-run notebooks as backup
- Offer office hours after workshop

---

## 📚 Additional Resources for Facilitators

### Deep Dive Topics
- MLflow advanced features
- Model Registry API
- Databricks Feature Store
- Model Serving endpoints
- CI/CD for ML models

### Extended Workshop Ideas
- **Day 2:** Advanced MLOps (monitoring, retraining, A/B testing)
- **Day 3:** Production deployment (REST APIs, integration patterns)
- **Workshop+:** Add deep learning module with TensorFlow/PyTorch

### Follow-up Materials
- Advanced notebook collection
- Video recordings
- Office hours schedule
- Slack/Teams channel for questions
- Monthly ML best practices sessions

---

## 🎯 Success Metrics

### Participant Success Indicators
- [ ] All notebooks run successfully
- [ ] Participants can explain MLflow benefits
- [ ] Understanding of Model Registry workflow
- [ ] Can compare models meaningfully
- [ ] Grasp when to use AutoML

### Workshop Success Metrics
- Post-workshop survey score > 4.5/5
- 90%+ completion rate
- Participants use skills in their projects
- Positive feedback on practical applicability
- Requests for advanced follow-up sessions

---

## 📝 Feedback Collection

### During Workshop
- Monitor chat for questions/confusion
- Watch participant screens for issues
- Check pace via quick polls
- Note common questions

### Post Workshop
- Send survey within 24 hours
- Request specific feedback on:
  - Content clarity
  - Pacing
  - Hands-on balance
  - Practical applicability
  - Suggested improvements

### Survey Questions
1. Rate overall workshop quality (1-5)
2. Was the pace appropriate?
3. Which module was most valuable?
4. Which module needs improvement?
5. Will you use these skills?
6. What topics should we add?
7. Any technical issues?
8. Would you recommend this workshop?

---

## 🌟 Best Practices

### Before Running Workshop
1. Run through all notebooks yourself
2. Time each section
3. Identify potential stumbling points
4. Prepare additional examples
5. Have backup materials ready

### During Workshop
1. Start on time
2. Stick to schedule (with flexibility)
3. Encourage questions
4. Monitor participant progress
5. Take short breaks

### After Workshop
1. Share all materials
2. Provide additional resources
3. Offer office hours
4. Collect feedback
5. Follow up on questions

---

## 📞 Support Contacts

### Technical Issues
- Databricks Support: [link to support]
- Internal IT Help: [your org contact]

### Content Questions
- Workshop Lead: [contact info]
- ML Team: [contact info]

---

**Good luck with your workshop! 🚀**

*Remember: The goal is not just to teach tools, but to empower data scientists to build better ML systems.*

