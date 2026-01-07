# DiPM: Decoupling and Recombining of Parameters in Low-Rank Adaptation for Module Ability Integration

<img width="623" height="412" alt="1767765801596" src="https://github.com/user-attachments/assets/701e4fc7-34e4-4c8a-a5ef-92331830c617" />

Overall framework of DiPM, including: decoupler, modulator, and recombiner.

## train -- train data
### multi-tasking: mnli.json / mrpc.json / cola.json / rte.json
### unlearning: alpaca.json / wizard.json / toxic.json
### transfer: neutral.json / republican.json

## test -- test data
### multi-tasking: mnli.json / mrpc.json / cola.json / rte.json
### unlearning: toxic_instruction.json
### transfer: neutral.json / republican.json

## evaluation -- evaluation procedure on the test set

## operators -- different defined operators

## train_param_setting -- parameter settings for model training using llama_factory
