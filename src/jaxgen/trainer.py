import jax
from jaxgen.model import BaseModel
from pydantic import BaseModel as PydanticBaseModel

class TrainerConfig(PydanticBaseModel):
    learning_rate: float
    epochs: int


class Trainer:
    def __init__(self, model: BaseModel, dataloader, config: TrainerConfig):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.state = self.model.create_train_state(jax.random.PRNGKey(0), config.learning_rate)

    def train(self):
        for epoch in range(self.config['epochs']):
            for batch in self.dataloader['train']:
                self.state, loss = self.model.train_step(self.state, batch)
                # TODO: Add logging, checkpointing, etc.
                print(f"Epoch {epoch}, Loss: {loss}")

    def evaluate(self):
        metrics = self.model.evaluate(self.state, self.dataloader['eval'])
        # Add logging, etc.
        return metrics

    def sample(self, rng, num_samples):
        samples = self.model.sample(rng, num_samples)
        return samples
