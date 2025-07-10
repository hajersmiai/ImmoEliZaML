## Feedback
- Excellent readme
- good to add the slides in the repo
- notebooks are messy, unused and commented blocks of code. Don't forget to clean and remove useless code
- code architecture is not clean, things are all over the place. Create logical folders
  - notebooks/ for exploration
  - model_scripts/ for models training
  - models_files/ if you same some (like the catboost.cbm)
  - utils/ for cleaning and preprocessing
  - ...
- Use .py files instead of notebooks (1 exception: EDA or data viz)
- Ok i think I understand that you've added all your code. This is not what was expected, you had to agree on 1 model, 1 piece of code to give the final result of the team effort. What you've done is really difficult to read and use. I have no idea where the best model is, how to run it,... And I won't search it should be easy to spot.
- I can see the chatgpt emojis and comments by the way. What have you done yourself?
- good use of merges but pay attention to your commits messages, it should be more informative. It's good you have merged the team work but next step would be to summarize it, standardize it, structure it,...
- Depending on the code, good files structure (1 file per model) but no classes or functions, improve your OOP logic in the future.

But even if there is room for improvement this is a complete project, you've done it :fire: Have this comments in mind in the future and it'll be all right

## Evaluation criteria

| Criteria       | Indicator                                     | Yes/No |
| -------------- | --------------------------------------------- | ------ |
| 1. Is complete | Know how to answer all the above questions.   | YES    |
|                | `pandas` and `matplotlib`/`seaborn` are used. | YES    |
|                | All the above steps were followed.            | YES    |
|                | A nice README is available.                   | YES    |
|                | Your model is able to predict something.      | YES    |
| 2. Is good     | You used typing and docstring.                | Not always    |
|                | Your code is formatted (PEP8 compliant).      | Not always    |
|                | No unused file/code is present.               | NO    |
