import sys
import click

from similarity import Similarity
from load import DataLoad


@click.command()
@click.option('-p', '--picture', type = click.STRING, default = '', help = "포켓몬 사진 이름 입력 ex) -p 'abomasnow.png'")
@click.option('-c', '--cos-sim', type = click.STRING, multiple = True, default = ['', ''],\
               help = "cos 유사도를 구할 두 가지 입력 ex) -c 'abomasnow.png' -c 'abra.png'")

def start_batch(picture, cos_sim):
    print('<<<< DATA LOAD >>>>')
    data_list = DataLoad()._load()
    print("데이터 개수 : ", len(data_list))

    if not data_list:         # if data_list is None:
        print(f"<<<< KILL BATCH : data_list == {data_list} >>>>")
        sys.exit(1)             # 시스템 강제 종료

    if cos_sim == ('', ''):
        print("##### SHOW SIMILARITY GRAPH #####")
        print(f"\tInput Value : picture == '{picture}' ")
        Similarity(picture, data_list).visualize()
        print("##### END #####")
        sys.exit(0)     # 정상적으로 종료
    elif picture == '':
        print("##### SHOW COS-SIMILARITY #####")
        print(f"\tInput Value : {cos_sim} ")
        Similarity(picture, data_list).cos_similarity(cos_sim)
        print("##### END #####")
        sys.exit(0)     # 정상적으로 종료

if __name__ == '__main__':
    start_batch()