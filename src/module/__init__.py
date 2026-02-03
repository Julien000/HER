

from module.OurTrainer import Our_ModelTrainer, TextModelrainer
def LODA_TRAINER( config, train_loader):
    model_maps={
        "mm_Ours": Our_ModelTrainer,
        "txt_Ours": TextModelrainer,
    }
    return model_maps[config.model](config, train_loader)