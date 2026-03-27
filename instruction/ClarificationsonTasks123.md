Please review the following guidelines carefully:



Task 1 (The Baseline): The goal here is simply to train your own nanoGPT model on the ROCStories training set. Because this is meant to be a basic task, you do not have to make improvements on data or architecture. Your goal is just to get the base model training correctly.



Task 2 (Exploration for Story Generation): While the assignment encourages exploration beyond the base setup and mentions trying different datasets like Q\&A, dialogue, or even SVG icons, please note that the core focus remains on story generation. The intention behind trying different task settings is to explore whether co-training on them might increase the model's story generation ability. Task 2 is not meant for exploring tasks completely unrelated to story generation.



Task 3 (Best Checkpoint Submission): For this task, you are required to submit your best checkpoint for story generation. You should not just submit the basic checkpoint from Task 1. We expect to see the best model you produced after applying the various improvements explored in Task 2.



Task 3 Constraints \& Allowances:



Datasets \& Size: You are allowed to use external datasets beyond ROCStories to improve your model, provided that your final model size does not exceed 32M.



Architecture \& Training: You are allowed to change your training method, and you may modify the architecture by including your own model.py file in your submission.



Evaluation Strictness: You cannot change the eval.py script or the upload script (hf\_upload.py / hf load.py ).



Pre-processing: Because the evaluation scripts must remain untouched, you are not allowed to change the pre-processing of the input text. Your final model must be perfectly loadable with your model.py and testable on our private dataset using the default text-preprocessing pipeline.



If you have any further questions about these constraints, please let me know!

