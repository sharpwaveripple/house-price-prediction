from data import make_dataset
from features import build_features
from models import eval_models, train_model, predict_model


make_dataset.main()
build_features.main()
eval_models.main()
train_model.main()
predict_model.main()
