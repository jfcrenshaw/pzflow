# Common gotchas

* It is important to note that there are two different conventions in the literature for the direction of the bijection in normalizing flows. PZFlow defines the bijection as the mapping from the data space to the latent space, and the inverse bijection as the mapping from the latent space to the data space. This distinction can be important when designing more complicated bijections (e.g., in Example 2 above).

* If you get NaNs during training, try decreasing the learning rate (the default is 1e-3):

```python
import optax

opt = optax.adam(learning_rate=...)
flow.train(..., optimizer=opt)
```
