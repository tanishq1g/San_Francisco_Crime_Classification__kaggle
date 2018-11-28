class Data_Postprocessing:
    def make_submission(self, predictions, labels, ids):
        result = pd.DataFrame(predicted, columns = labels)
        result.insert(loc = 0, column = 'Id', value = ids)
        result.to_csv('submission.csv', index = False)