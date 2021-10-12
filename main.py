from embedding import data_embedding
#from vanillaRL import ddpg_model
from vanillaRL2 import ddpg_model2
#from result_test import rank_result
def main(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


    #data_embedding()
    ddpg_model2()
    #ddpg_model()
    #print(rank_result())

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm what')
