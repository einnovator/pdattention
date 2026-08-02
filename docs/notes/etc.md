change the nb to train with syntetic dataset not just 3 examples but with several sizes..put each notebook cells code inside functions for reusablity..then for the default size create cells jsut after each function defining cell to run for the DEFAULT_SIZE=3 then at the end of notebooks att cells and md section to train to run the worklaod for additional sizes 30,, 300, 3000, 30000.
See if adjustinng number the model parameter size, nunber of ephocs is required to make the model have a good performance

--


create util functions file to include function reusable across notebooks extracted from D:\git\rd\pdattention\nb\pra_standalone.ipynb if they can be eused to train model with different dataset
then creae a notbook with same worflow one per datase under data/
create also notebook for the syntetic dataset but using the utils to compare for parity with pra_standalone
name of nb can be pra_train_eval_<dataset>.ipynb
run each notebook for each dataset.