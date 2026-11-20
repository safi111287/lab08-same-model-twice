# Framework comparison for Lakeshore Bike Share

I recommend keeping PyTorch for the weekly bike demand job. The existing loop
already works, and the project has tests for gradient handling and restoration
of the best weights. For a two-person team, keeping that implementation avoids
maintaining two training paths. The teammate who knows Keras can still review
the shared data preparation and use the epoch CSV to monitor the weekly runs.

The error difference was small. Across three seeds, test MAE averaged
2.509 rentals for PyTorch and 2.462 for Keras, with sample
standard deviations of 0.082 and 0.044.
PyTorch had the lower test error at seed 2026; Keras had the lower error at the
other two seeds. The advantage did not hold across all runs, so these results
are too limited to choose a framework on prediction error alone.

The runs averaged 12.0 seconds for PyTorch and 18.4 seconds
for Keras on the same CPU. That is elapsed time for these scripts, including
Keras checkpoint writes, rather than a general speed comparison. A few seconds
would make little difference for a job that runs once a week. The code is more
relevant to maintenance: Keras uses one `fit` call and two callbacks, while
the PyTorch loop explicitly clears gradients, backpropagates and copies the best
weights. Choosing PyTorch means keeping responsibility for those steps and giving
up the ready-made Keras checkpoint workflow.

Both runs used the same data, epoch budget, batch size, loss and Adam settings.
The frameworks use different initializers, shuffle algorithms and
[Adam epsilon conventions](https://keras.io/api/optimizers/adam/), so matching
seeds does not make the training trajectories identical. The weight-transfer
test verifies equivalence independently of training.

PyTorch also aligns with the
[Hugging Face Transformers workflow](https://huggingface.co/docs/transformers/main/installation)
if the team later tries a compatible pretrained model. I would switch to Keras if
the weekly job needed to fit an existing Keras deployment and monitoring workflow
that both teammates could support.
