import click

from two_step_verification.sd import Dreamer, TextDreamer
from two_step_verification.tiny_sam import TinySamEverything, TinySamPoint
from two_step_verification.webcam import stream


@click.command()
@click.option(
    "--method",
    type=click.Choice(["dream", "text-dream", "sam-everywhere", "sam-point"]),
    default="dream",
)
def main(method):
    if method == "dream":
        callable = Dreamer()
    elif method == "text-dream":
        callable = TextDreamer()
    elif method == "sam-point":
        callable = TinySamPoint()
    else:
        callable = TinySamEverything()
    stream(callable)


if __name__ == "__main__":
    main()
