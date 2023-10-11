
import argparse
from modelAPI import ModelAPI
from transformers import TrainingArguments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--output_dir', type=str, default="./output", help='Output directory.')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Train batch size.')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate.')
    args = parser.parse_args()
    
    model_api = ModelAPI(model_name=args.model_path, data_path=args.data_path, output_dir=args.output_dir)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_dir='./logs'
    )
    
    # Prepare data
    train_dataset = model_api.prepare_datasets(start_idx=0, end_idx=10000)  # Adjust indices as needed
    
    # Train model
    model_api.train_model(train_dataset, training_args)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for progressive training.")
    
    args = parser.parse_args()
    main(args)
