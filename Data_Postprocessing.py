import pickle
import pandas as pd

class Data_Postprocessing:
    def make_submission(self, predictions, labels, ids):
        result = pd.DataFrame(predictions, columns = labels)
        result.insert(loc = 0, column = 'Id', value = ids)
        result.to_csv('submission.csv', index = False)

    def make_pickle(self, model, picklename = "model.pkl"):
        model_pickle = open(picklename, 'wb')
        pickle.dump(model, model_pickle)
        model_pickle.close()
        return model_pickle

    def use_pickle(self, pickle_file, Xtest):
        model = open(pickle_file, 'rb')
        model_pickle = pickle.load(model)
        print("Loaded model :: ", model_pickle)
        predictions = model_pickle.predict(Xtest)
        return predictions