
from collections import namedtuple
from src import bootstrap
import settings
import const

if __name__ == "__main__":

    argsClass = namedtuple('argsClass', 'build predict')
    buildClass = namedtuple('argsClass', 'input directed sample method dimension windowsize walklen nbofwalks embedtype classificationfunc optimizeclassifier '
                                         'temp_dir temp_id logfile train_ratio verbose keep_dropout use_cuda epoch_num batch_size task force')
    print(const.SLASHDOT_GRAPH)
    build = buildClass(input=settings.config[const.SLASHDOT_GRAPH],
                       directed=True, sample=["degree", 120], method="3type",
                       dimension=10, windowsize=3, walklen=50, nbofwalks=20, embedtype="py", classificationfunc= "MLP", optimizeclassifier= True,
                       temp_dir=settings.config[const.TEMP_DIR],
                       temp_id="slash-full", train_ratio=0.8, verbose=True, logfile = "log.txt", keep_dropout = 0.8, use_cuda=False, epoch_num=10,
                       batch_size = 512, task = 'link',
                       #force=['model'])
                       force=[ 'sample', 'preprocess', 'postprocess', 'model'])

    # args = argsClass(build=build, predict=None)
    # bootstrap.main(args)
    print("----------------------------")
    # build = build._replace(method="attention")
    args = argsClass(build=build, predict=None)
    bootstrap.main(args)
    print("----------------------------")

