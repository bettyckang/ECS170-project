'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting


class Setting_Train_Only(setting):
    # the joke dataset has no test split: the model trains on every joke, and its generated jokes
    # are then compared with those same training jokes

    def load_run_save_evaluate(self):

        # load dataset
        loaded_data = self.dataset.load()

        # run MethodModule (train, then generate)
        self.method.data = loaded_data
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        return self.evaluate.evaluate()
