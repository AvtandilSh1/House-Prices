# House Prices — Advanced Regression Techniques

კაგლის კონკურსია სადაც სახლების ფასი უნდა ვიწინასწარმეტყველოთ. მონაცემებში 79 სხვადასხვა მახასიათებელია — ფართობი, ხარისხი, სარემონტო ისტორია და სხვა. შეფასება ხდება RMSE-ით.

---

## რა გავაკეთე

პირველ რიგში მონაცემები გავასუფთავე, შემდეგ ახალი ფიჩერები შევქმენი, საჭირო სვეტები ავარჩიე და სხვადასხვა მოდელები გავტესტე. ყველა ექსპერიმენტი MLflow-ზე დავლოგე DagsHub-ის საშუალებით.

---

## ფაილები

- `model_experiment.ipynb` — ძირითადი ნოუთბუქი, აქ ყველაფერია
- `model_inference.ipynb` — საუკეთესო მოდელით ფასების პროგნოზი
- `submission.csv` — კაგლზე ასატვირთი შედეგი
- `data/train.csv`, `data/test.csv` — კაგლის მონაცემები

---

## Cleaning

ორი ჩანაწერი ამოვიღე სადაც სახლი ძალიან დიდი იყო მაგრამ ფასი ძალიან დაბალი — Outlier-ები იყვნენ და მოდელს უფუჭებდნენ.

NaN-ები შევავსე სტრატეგიის მიხედვით:
- სადაც არარსებობას ნიშნავდა — "None" ან 0
- LotFrontage — Neighborhood-ის Median
- დანარჩენი — Mode ან Median

---

## Feature Engineering

შევქმენი რამდენიმე ახალი სვეტი:

- `TotalSF` — სახლის საერთო ფართობი
- `TotalBath` — ყველა სველი წერტილი ერთად
- `HouseAge`, `RemodelAge` — სახლის ასაკი და ბოლო რემონტის ასაკი
- `HasPool`, `HasGarage`, `HasBsmt`, `Has2ndFloor` — არის თუ არა (0 ან 1)
- `OverallQual_TotalSF` — ხარისხი × ფართობი

კატეგორიული სვეტები: ხარისხის სვეტები რიცხვებად (1-5), დანარჩენი One-Hot Encoding-ით. SalePrice — log გარდაქმნა, skewed სვეტები — Box-Cox.

---

## Feature Selection

სამი მიდგომა გამოვიყენე:

1. კორელაცია SalePrice-თან (|r| > 0.1)
2. Lasso — ნულოვანი კოეფიციენტის მქონე სვეტები ამოვაგდე
3. Random Forest Importance

საბოლოოდ სამივეს Union ავიღე.

---

## Training

**Linear Regression** — Baseline. CV RMSE = 0.10943

**Ridge** — L2 regularization, 5 სხვადასხვა alpha:
- alpha=0.001 → regularization პრაქტიკულად არ მუშაობს, LR-ის იდენტური
- alpha=10, 50 → Underfitting იწყება
- alpha=500, 5000 → ძლიერი Underfitting (CV RMSE = 0.19)

**Lasso** — L1 regularization, 4 alpha:
- alpha=0.0001 → **საუკეთესო შედეგი, CV RMSE = 0.10930**
- alpha=0.001 და მეტი → Underfitting

**Decision Tree** — 4 სხვადასხვა max_depth:
- depth=2 → Underfitting, ხე ძალიან მარტივია
- depth=5 → Overfitting იწყება
- depth=10 → ძლიერი Overfitting (Train RMSE = 0.058, CV RMSE = 0.190)
- depth=None → Extreme Overfitting, Train RMSE = 0.001 — ყველა ნიმუში ზეპირად

---

## საუკეთესო მოდელი

**Lasso alpha=0.0001**, CV RMSE = 0.10930

ეს მოდელი MLflow Model Registry-ში შეინახა სახელით `HousePricesModel`. `model_inference.ipynb` ამ მოდელს პირდაპირ Registry-დან ჩამოტვირთავს და test set-ზე გაუშვებს.

---

## MLflow

ექსპერიმენტები: https://dagshub.com/ashos22/House-Prices.mlflow

თითოეულ run-ზე დავლოგე: `cv_rmse`, `cv_std`, `train_rmse`, `train_r2`, `overfit_gap`
