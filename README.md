# 데이터사이언스 프로젝트 2
<hr>

## 1. 목표
[kaggle의 Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) 데이터를 활용해 연체 가능성을 예측하고, 대출 적격 여부를 판단한다.

<br>

## 2. 데이터 설명
Home Credit Default Risk 데이터는 체코에서 설립되어 네덜란드에 본사를 두고 있는 Home Credit 사가 제공한 것이다. Home Credit 사는 비은행 금융 기관으로서, 신용 이력이 적은 사람들을 주요 고객으로 하고 있다. CIS(소련 붕괴 후 구성된 독립 국가 연합)와 SEA countries(Southeast Asian Countries, 동남아시아)에서 수집된 자료다. (Kaz, Russia, Vietnam, China, Indonesia, Phillipines 등)

Bureau data는 은행권 경험이 적은 사람들의 자료이기 때문에 대출 시 신용도를 판단하기에 충분한 데이터는 아니다. 반면 현재 대출과 과거 대출 자료가 신용도를 판단하기에 보다 유의미하다.

*Home Credit 사의 주요 상품*
- Revolving loan (credit card)  
리볼빙 대출은 해당 월에 지급해야 할 신용카드 대금을 익월로 일부 미루는 대신 수수료(이자)를 부담하는 상품이다.

- Consumer installment loan (Point of sales loan – POS loan)  
소비재 구매 시 할부 대출을 의미한다.  
예컨대 `previous_application.csv`의 14441번 사람(`SK_ID_CURR = 104895`)은 할부 대출 서비스를 이용했다. 구매한 제품의 가격은 US$ 21,442.5<span style="color:gray">(달러라고 가정함)</span>이기 때문에 대출 신청액(`AMT_APPLICATION`)과 대출 승인액(`AMT_CREDIT`)은 상품의 가격인 US$ 21,442.5이다. 계약금(`AMT_DOWN_PAYMENT`)은 US$ 0으로 계약금이 없는 것을 볼 때 상대적으로 신용도가 높다고 판단할 수 있다. 왜냐하면 신용도가 낮을 경우 할부 대출 시 초기 계약금의 비율(`RATE_DOWN_PAYMENT`)를 높게 설정하여 디폴트가 발생해도 손실 금액을 최소화하기 때문이다.  
한편 이 계약의 기본 이자율(`RATE_INTEREST_PRIMARY`)은 19.69%이나, 이자 할증율(`RATE_INTEREST_PRIVILEGED`)이 86.7336%이다. 따라서 총 이자율은 약 연 36.77%(19.69 * (1 + 0.867336))다.  
추가로 14441번 사람이 상품을 구매한 시점(즉, 리볼빙론 계약을 신청한 시점)은 일요일(`WEEKDAY_APPR_PROCESS_START`) AM 10시(`HOUR_APPR_PROCESS_START`)다.

- Installment cash loan
현금 서비스를 의미한다. 신용 대출보다 리스크가 높으나, Home Credit 사의 주요 상품이다.

<br>

[참고자료](https://www.kaggle.com/c/home-credit-default-risk/discussion/63032)

# feature 설명

- `AMT_INSTALMENT`/`AMT_ANNUITY`: 원금과 이자를 포함한 월 상환액

- CNT*: count
- AMT*: amount, 일반적으로 화폐와 관련된 수치임
- FLAG: 1 = yes, 0 = No
- FLAG_DOCUMENT: 대출 신청 시 소득 증명서 등을 제출했는지 여부
- OBS*: observation, 발생할 가능성이 있음.
- DEF*: actual default, 실제로 발생함.

- x-sell: 고객이 이전 대출에서 리스크 평가를 이미 받음.
- walk-in: 신규 고객

## 2. 프로세스
1. 8개 datasets 결측치 및 이상치 처리
2. Encoding / Scaling

## 참고자료(data)
[Host의 feature 설명 댓글](https://www.kaggle.com/c/home-credit-default-risk/discussion/57054)

[Feature 해석](https://chocoffee20.tistory.com/6)

[참고자료 1](https://medium.com/mighty-data-science-bootcamp/kaggle-도전기-home-credit-default-risk-part-1-735030d40ee0)

[참고자료 2](https://john-analyst.medium.com/캐글-home-credit-default-risk-9225050b6fa6)

[참고자료 3](https://velog.io/@fiifa92/첫-번째-모델-학습-및-성능-평가)

[참고자료 4](https://suhyun-cho.github.io/kaggle/kaggle-HomeCredit-default-risk-eda-and-FeatureEngineering_beginner/)


<br>

## 참고자료(머신러닝)
[SAS BLOG](https://www.sas.com/ko_kr/solutions/ai-mic/blog/machine-learning-algorithm-cheat-sheet.html)

[YSY의 데이터 분석 블로그](https://ysyblog.tistory.com/category/Machine%20Learning)

[테디노트](https://teddylee777.github.io/categories/scikit-learn/)

[Collect all discussion in Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/discussion/60521)

<br>

## 참고자료(code)
[LightGBM](https://www.kaggle.com/jsaguiar/lightgbm-7th-place-solution)

[Base Model with +0.804 AUC on Home Credit](https://www.kaggle.com/hikmetsezen/base-model-with-0-804-auc-on-home-credit)

[-](https://www.kaggle.com/qbxkvbf/bigdata-project-eda-fe-qbxkvbf5)

[-](https://www.kaggle.com/mathchi/home-credit-risk-with-detailed-feature-engineering)

[open-solution-home-credit](https://github.com/minerva-ml/open-solution-home-credit)

[-](https://www.kaggle.com/oriroval/naya-classification-project-4-ori-and-ori)

[](https://www.kaggle.com/shailaja4247/tackle-any-credit-risk-analysis-problem-homecredit#CatBoost_clf=CatBoostRegressor(iterations=50,-depth=3,-learning_rate=0.1,-loss_function='RMSE'))

# 참고자료
[A Gentle Introduction](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction)

[Intro to Model Tuning: Grid and Random Search](https://www.kaggle.com/willkoehrsen/intro-to-model-tuning-grid-and-random-search)

[Feature Engineering using Feature Tools](https://www.kaggle.com/willkoehrsen/feature-engineering-using-feature-tools)

[Clean Manual Feature Engineering](https://www.kaggle.com/willkoehrsen/clean-manual-feature-engineering)

[Introduction to Manual Feature Engineering](https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering)

[Introduction to Manual Feature Engineering P2](https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering-p2)

[Automated Feature Engineering Basics](https://www.kaggle.com/willkoehrsen/automated-feature-engineering-basics)

[Tuning Automated Feature Engineering (Exploratory)](https://www.kaggle.com/willkoehrsen/tuning-automated-feature-engineering-exploratory)

[Feature Selection](https://www.kaggle.com/willkoehrsen/introduction-to-feature-selection)

[Model Tuning Results: Random vs Bayesian Opt](https://www.kaggle.com/willkoehrsen/model-tuning-results-random-vs-bayesian-opt)

[Automated Model Tuning](https://www.kaggle.com/willkoehrsen/automated-model-tuning)


[EDA](https://www.kaggle.com/codename007/home-credit-complete-eda-feature-importance)


<br>

## XGBoost
<hr>

[XGBoost 사용하기 by 소고(KR)](https://brunch.co.kr/@snobberys/137)

[XGBoost(KR)](https://dining-developer.tistory.com/3)

[XGBoost Hyper Parameter 설명(KR)](http://machinelearningkorea.com/2019/09/29/lightgbm-파라미터/)

[XGBoost Hyper Parameter 공식문서](https://xgboost.readthedocs.io/en/latest/parameter.html)

[Xgboost 하이퍼 파라미터 튜닝(KR)](https://www.kaggle.com/lifesailor/xgboost)


<br>

## Reference Code
[LGBM_FULL(EN)](https://www.kaggle.com/chienhsianghung/home-credit-default-risk-lgbm-w-domain-fts)

[EDA, Feature Engineering(LGBM_FULL 참고(KR))](https://www.kaggle.com/whtngus4759/eda-and-feature-engineering-for-beginner#10\)-Model-Interpretation:-Feature-Importances)


<br>

## CatBoost
Category features를 Encoding할 필요 없이, 학습 과정에서 자동으로 변환함.

[Official Website](https://catboost.ai)

[What’s so special about CatBoost?[EN]](https://hanishrohit.medium.com/whats-so-special-about-catboost-335d64d754ae)

[CatBoost(KR)](https://gentlej90.tistory.com/100)


<br>

## Bayesian Hyper Parameter Optimization 
[A Conceptual Explanation of Bayesian Hyperparameter Optimization for Machine Learning](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f)

[베이지안 최적화에 기반한 HyperOpt를 활용한 하이퍼 파라미터 튜닝](https://teddylee777.github.io/thoughts/hyper-opt)