import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import sys
if (_package_root := str(Path(__file__).absolute().parents[2])) not in sys.path:
    sys.path.insert(0, _package_root)

import click


@click.group(help='SCOPE command line interface.')
def cli():
    pass


def _make_argparse_command(parser, runner):
    @click.command(
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
        help=parser.description,
        add_help_option=False,
    )
    @click.pass_context
    def command(ctx):
        argv = list(ctx.args)
        if any(token in {"-h", "--help"} for token in argv):
            click.echo(parser.format_help())
            return
        runner(argv)
    return command


def main():
    from scope.scripts import infer_video, train
    cli.add_command(_make_argparse_command(infer_video.build_parser(), infer_video.main), name='infer')
    cli.add_command(_make_argparse_command(infer_video.build_parser(), infer_video.main), name='infer_video')
    cli.add_command(train.main, name='train')
    cli()


if __name__ == '__main__':
    main()
