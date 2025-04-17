import optuna

def create_study(name, direction="minimize"):
    return optuna.create_study(study_name=name, direction=direction)