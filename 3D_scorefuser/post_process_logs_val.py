import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description='print val loss curve from slurm logs')
    parser.add_argument('--log', type=str, required=True, help='slurm log file')
    args = parser.parse_args()

    # read in log files
    with open(args.log, 'r') as f:
        lines = f.readlines()
        val_loss = []
        for i in range(len(lines)):
            if 'val loss epoch at milestone' in lines[i]:
                # find the val loss after :
                mile_stone, cur_val_loss = lines[i].split(':')[0], lines[i].split(':')[1]
                cur_val_loss = float(cur_val_loss[:8])
                mile_stone = int(mile_stone.split('milestone')[-1])
                val_loss.append((mile_stone, cur_val_loss))
        # calculate avg val loss for each milestone
        avg_val_loss = {}
        # std_val_loss = {}
        for i in range(len(val_loss)):
            mile_stone, cur_val_loss = val_loss[i]
            if mile_stone not in avg_val_loss:
                avg_val_loss[mile_stone] = [cur_val_loss]
            else:
                avg_val_loss[mile_stone].append(cur_val_loss)
        for mile_stone in avg_val_loss:
            avg_val_loss[mile_stone] = np.mean(avg_val_loss[mile_stone]), np.std(avg_val_loss[mile_stone]), avg_val_loss[mile_stone]
            
        for i in range(len(avg_val_loss.keys())):
            print("In epoch "+str(i)+", val_loss mean:",format(avg_val_loss[i][0],".5f"), " val_loss std:", format(avg_val_loss[i][1],".5f"), "sampled loss", avg_val_loss[i][2])
        # print(avg_val_loss)


if __name__ == '__main__':
    main()



