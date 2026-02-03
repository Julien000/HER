import datetime
import os

class SimpleLogger:
    """
    Simple logger that writes to both console and file
    """
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        # Create directory if not exists
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        # Create or clear the log file
        with open(self.log_file_path, 'w') as f:
            f.write(f"Training log started at {datetime.datetime.now()}\n")
            f.write("="*60 + "\n")
    
    def log(self, message):
        """Log a message with timestamp"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)  # Also print to console
        
        # Write to log file
        with open(self.log_file_path, 'a') as f:
            f.write(log_message + "\n")
    
    def log_separator(self):
        """Add a separator line to the log"""
        separator = "-" * 40
        with open(self.log_file_path, 'a') as f:
            f.write(separator + "\n")

    def log_config(self, config):
        """Log configuration details"""
        self.log("Configuration Details:")
        self.log("-" * 30)
        config_dict = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
        for key, value in config_dict.items():
            self.log(f"  {key}: {value}")
        self.log("-" * 30)

