import click
import gin
from pipelines import ModelTrainPipeline

def run_pipeline(pipeline):

    pipeline.train()

@click.group()
def cli():
    pass



@click.command()
@click.option('--config_path', required=True, help='Model Name')
def train(config_path):

    gin.parse_config_file(config_path)

    pipeline = ModelTrainPipeline()
    pipeline.run()

@click.command()
def inference():
    click.echo('Running Inference Step')



cli.add_command(train)
cli.add_command(inference)


if __name__ == '__main__':
    cli()