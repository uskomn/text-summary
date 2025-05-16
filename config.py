class Config:
    cache_dir='./cache'
    model_name = "facebook/bart-large-cnn"
    train_batch_size = 2
    eval_batch_size = 4
    gradient_accumulation_steps = 8
    num_train_epochs = 3
    learning_rate = 3e-5
    max_input_length = 1024
    max_target_length = 128
    label_smoothing = 0.1
    output_dir = "./model"
