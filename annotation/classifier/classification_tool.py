"""
 One-click image sorting/labelling script. Copies or moves images from a folder into subfolders. 
 This script launches a GUI which displays one image after the other and lets the user give different labels
 from a list provided as input to the script. In contrast to original version, version 2 allows for 
 relabelling and keeping track of the labels.
 Provides also short-cuts - press "1" to put into "label 1", press "2" to put into "label 2" a.s.o.

 USAGE:
 run 'python sort_folder_vers2.py' or copy the script in a jupyter notebook and run then

 you need also to provide your specific input (source folder, labels and other) in the preamble
 original Author: Christian Baumgartner (c.baumgartner@imperial.ac.uk)
 changes, version 2: Nestor Arsenov (nestorarsenov_AT_gmail_DOT_com)
 Date: 24. Dec 2018
"""

import pandas as pd
import json
import argparse
import tkinter as tk
import os
from shutil import copyfile, move
from PIL import ImageTk, Image
from utils.general_utlis import read_txt, save_txt

# Define global variables, which are to be changed by user:

# added in version 2
global exception_paths, input_folder, paths, train_val_info_folder

# the folder in which the pictures that are to be sorted are stored
# don't forget to end it with the sign '/' !
input_folder = '/file_path/to/image_folder/'

# the different folders into which you want to sort the images, e.g. ['cars', 'bikes', 'cats', 'horses', 'shoes']
labels = ["label1", "label2", "label3"]

# provide either 'copy' or 'move', depending how you want to sort the images into the new folders
# - 'move' starts where you left off last time sorting, no 'go to #pic', works with number-buttons for labeling, no txt-file for tracking after closing GUI, saves memory
# - 'copy' starts always at beginning, has 'go to #pic', doesn't work with number-buttons, has a txt-for tracking the labels after closing the GUI
copy_or_move = 'copy'

# Only relevant if copy_or_move = 'copy', else ignored
# A file-path to a txt-file, that WILL be created by the script. The results of the sorting wil be stored there.
# Don't provide a filepath to an empty file, provide to a non-existing one!
# If you provide a path to file that already exists, than this file will be used for keeping track of the storing.
# This means: 1st time you run this script and such a file doesn't exist the file will be created and populated,
# 2nd time you run the same script, and you use the same df_path, the script will use the file to continue the sorting.
df_path = 'non_existing_file_df.txt'

output_file_names_path = 'final_paths.txt'

output_file_names = []

# a selection of what file-types to be sorted, anything else will be excluded
file_extensions = ['.jpg', '.png', '.whatever']

# set resize to True to resize image keeping same aspect ratio
# set resize to False to display original image
resize = True


class ImageGui:
    """
    GUI for iFind1 image sorting. This draws the GUI and handles all the events.
    Useful, for sorting views into sub views or for removing outliers from the data.
    """

    def __init__(self, master, labels, paths, is_default):
        """
        Initialise GUI
        :param master: The parent window
        :param labels: A list of labels that are associated with the images
        :param paths: A list of file paths to images
        :return:
        """

        # Initialize the mirror count
        self.mirror_count = 0

        # So we can quit the window from within the functions
        self.master = master

        # Extract the frame so we can draw stuff on it
        frame = tk.Frame(master)

        # Initialise grid
        frame.grid()

        # Initialize default setting
        self.is_default = is_default
        # Start at the first file name
        self.index = 0
        self.paths = paths
        self.labels = labels

        #### added in version 2
        self.sorting_label = 'unsorted'
        ####

        # Number of labels and paths
        self.n_labels = len(labels)
        self.n_paths = len(paths)

        # Set empty image container
        self.image_raw = None
        self.image = None
        self.image_panel = tk.Label(frame)

        # set image container to first image
        self.set_image(paths[self.index])

        # Make buttons
        self.buttons = []
        for label in labels:
            self.buttons.append(
                    tk.Button(frame, text=label, height=2, fg='blue', command=lambda l=label: self.vote(l))
            )
            
        ### added in version 2
        self.buttons.append(tk.Button(frame, text="prev im", height=1, fg="green", command=lambda l=label: self.move_prev_image()))
        self.buttons.append(tk.Button(frame, text="next im", height=1, fg='green', command=lambda l=label: self.move_next_image()))
        self.buttons.append(tk.Button(frame, text="mirror count +", height=2, fg='red',
                                      command=lambda l=label: self.mirror_count_incr()))
        self.buttons.append(tk.Button(frame, text="mirror count -", height=2, fg='red',
                                      command=lambda l=label: self.mirror_count_decr()))
        ###

        # Add mirror count label
        self.mirror_count_label = tk.Label(frame, text="Mirror count: %d" % self.mirror_count)
        self.mirror_count_label.grid(row=5, column=2, sticky='we')
        
        # Add progress label
        progress_string = "%d/%d" % (self.index+1, self.n_paths)
        self.progress_label = tk.Label(frame, text=progress_string)
        
        # Place buttons in grid
        for ll, button in enumerate(self.buttons[:-4]):
            button.grid(row=0, column=ll, sticky='we')
        self.buttons[-4].grid(row=1, column=0, sticky='we')
        self.buttons[-3].grid(row=1, column=1, sticky='we')
        self.buttons[-2].grid(row=5, column=0, sticky='we')
        self.buttons[-1].grid(row=5, column=1, sticky='we')
            #frame.grid_columnconfigure(ll, weight=1)

        # Place progress label in grid
        self.progress_label.grid(row=0, column=self.n_labels, sticky='we') # +2, since progress_label is placed after
                                                                            # and the additional 2 buttons "next im", "prev im"
            
        #### added in version 2
        # Add sorting label
        sorting_string = os.path.split(df.sorted_in_folder[self.index])[-2]
        self.sorting_label = tk.Label(frame, text=("in folder: %s" % (sorting_string)))
        
        # Place typing input in grid, in case the mode is 'copy'
        if copy_or_move == 'copy':
            tk.Label(frame, text="go to #pic:").grid(row=2, column=0)

            self.return_ = tk.IntVar() # return_-> self.index
            self.return_entry = tk.Entry(frame, width=6, textvariable=self.return_)
            self.return_entry.grid(row=2, column=1, sticky='we')
            master.bind('<Return>', self.num_pic_type)
        ####
        
        # Place sorting label in grid
        self.sorting_label.grid(row=4, column=0, sticky='we') # +2, since progress_label is placed after
                                                                            # and the additional 2 buttons "next im", "prev im"
        # Place the image in grid
        self.image_panel.grid(row=3, column=0, columnspan=self.n_labels+1, sticky='we')

        # key bindings (so number pad can be used as shortcut)
        # make it not work for 'copy', so there is no conflict between typing a picture to go to and choosing a label with a number-key
        if copy_or_move == 'move':
            for key in range(self.n_labels):
                master.bind(str(key+1), self.vote_key)

    def show_next_image(self):
        """
        Displays the next image in the paths list and updates the progress display
        """
        self.index += 1
        progress_string = "%d/%d" % (self.index+1, self.n_paths)
        self.progress_label.configure(text=progress_string)

        # added in version 2
        sorting_string = os.path.split(df.sorted_in_folder[self.index])[-2] #shows the last folder in the filepath before the file
        self.sorting_label.configure(text=("in folder: %s" % (sorting_string)))
        ####

        if self.index < self.n_paths:
            self.set_image(df.sorted_in_folder[self.index])
        else:
            self.master.quit()

    # added in version 2
    def move_prev_image(self):
        """
        Displays the prev image in the paths list AFTER BUTTON CLICK,
        doesn't update the progress display
        """
        if self.index == 0:
            return
        self.index -= 1
        progress_string = "%d/%d" % (self.index+1, self.n_paths)
        self.progress_label.configure(text=progress_string)
        
        sorting_string = os.path.split(df.sorted_in_folder[self.index])[-2] #shows the last folder in the filepath before the file
        self.sorting_label.configure(text=("in folder: %s" % (sorting_string)))
        
        if self.index < self.n_paths:
            self.set_image(df.sorted_in_folder[self.index]) # change path to be out of df
        else:
            self.master.quit()
    
    # added in version 2
    def move_next_image(self):
        """
        Displays the next image in the paths list AFTER BUTTON CLICK,
        doesn't update the progress display
        """
        if self.index == self.n_paths:
            return
        self.index += 1
        progress_string = "%d/%d" % (self.index+1, self.n_paths)
        self.progress_label.configure(text=progress_string)
        sorting_string = os.path.split(df.sorted_in_folder[self.index])[-2] # shows the last folder in the filepath before the file
        self.sorting_label.configure(text=("in folder: %s" % (sorting_string)))
        
        if self.index < self.n_paths:
            self.set_image(df.sorted_in_folder[self.index])
        else:
            self.master.quit()

    def mirror_count_incr(self):
        if self.mirror_count >= self.n_paths:
            return
        self.mirror_count += 1
        self.mirror_count_label.configure(text="Mirror count: %d" % self.mirror_count)

    def mirror_count_decr(self):
        if self.mirror_count <= 0:
            return
        self.mirror_count -= 1
        self.mirror_count_label.configure(text="Mirror count: %d" % self.mirror_count)

    def set_image(self, path):
        """
        Helper function which sets a new image in the image view
        :param path: path to that image
        """
        image = self._load_image(path)
        self.image_raw = image
        self.image = ImageTk.PhotoImage(image)
        self.image_panel.configure(image=self.image)

    def vote(self, label):
        """
        Processes a vote for a label: Initiates the file copying and shows the next image
        :param label: The label that the user voted for
        """
        # added in version 2
        # check if image has already been sorted (sorted_in_folder != 0)
        if df.sorted_in_folder[self.index] != df.im_path[self.index]:
            # if yes, use as input_path the current location of the image
            input_path = df.sorted_in_folder[self.index]
            root_ext, file_name = os.path.split(input_path)
            root, _ = os.path.split(root_ext)
        else:
            # if image hasn't been sorted use initial location of image
            input_path = df.im_path[self.index]
            root, file_name = os.path.split(input_path)
        
        # input_path = self.paths[self.index]
        if copy_or_move == 'copy':
            self._copy_image(label, self.index)
        if copy_or_move == 'move':
            self._move_image(label, self.index)
        if self.is_default:
            update_annotation_info()    
        self.show_next_image()

    def vote_key(self, event):
        """
        Processes voting via the number key bindings.
        :param event: The event contains information about which key was pressed
        """
        pressed_key = int(event.char)
        label = self.labels[pressed_key-1]
        self.vote(label)
    
    # added in version 2
    def num_pic_type(self, event):
        """Function that allows for typing to what picture the user wants to go.
        Works only in mode 'copy'."""
        # -1 in line below, because we want images bo be counted from 1 on, not from 0
        self.index = self.return_.get() - 1
        
        progress_string = "%d/%d" % (self.index+1, self.n_paths)
        self.progress_label.configure(text=progress_string)
        sorting_string = os.path.split(df.sorted_in_folder[self.index])[-2] #shows the last folder in the filepath before the file
        self.sorting_label.configure(text=("in folder: %s" % (sorting_string)))
        
        self.set_image(df.sorted_in_folder[self.index])

    @staticmethod
    def _load_image(path):
        """
        Loads and resizes an image from a given path using the Pillow library
        :param path: Path to image
        :return: Resized or original image 
        """
        image = Image.open(path)
        if(resize):
            max_height = 500
            img = image 
            s = img.size
            ratio = max_height / s[1]
            image = img.resize((int(s[0]*ratio), int(s[1]*ratio)), Image.ANTIALIAS)
        return image

    @staticmethod
    def _copy_image(label, ind):
        """
        Copies a file to a new label folder using the shutil library. The file will be copied into a
        subdirectory called label in the input folder.
        :param input_path: Path of the original image
        :param label: The label
        """
        root, file_name = os.path.split(df.sorted_in_folder[ind])
        # two lines below check if the filepath contains as an ending a folder with the name of one of the labels
        # if so, this folder is being cut out of the path
        if os.path.split(root)[1] in labels:
            root = os.path.split(root)[0]
            os.remove(df.sorted_in_folder[ind])

        # Perform renaming
        if root != "output":
            file_name = root.split("/")[-2] + '_' + file_name

        output_path = os.path.join("output", label, file_name)

        if file_name not in output_file_names:
            output_file_names.append(file_name)

        copyfile(df.im_path[ind], output_path)

        # keep track that the image location has been changed by putting the new location-path in sorted_in_folder

        df.loc[ind, 'sorted_in_folder'] = output_path
        
        df.to_csv(df_path)

    @staticmethod
    def _move_image(label, ind):
        """
        Moves a file to a new label folder using the shutil library. The file will be moved into a
        subdirectory called label in the input folder. This is an alternative to _copy_image, which is not
        yet used, function would need to be replaced above.
        :param input_path: Path of the original image
        :param label: The label
        """
        root, file_name = os.path.split(df.sorted_in_folder[ind])
        # two lines below check if the filepath contains as an ending a folder with the name of one of the labels
        # if so, this folder is being cut out of the path
        if os.path.split(root)[1] in labels:
            root = os.path.split(root)[0]
        output_path = os.path.join(root, label, file_name)

        move(df.sorted_in_folder[ind], output_path)
            
        # keep track that the image location has been changed by putting the new location-path in sorted_in_folder

        df.loc[ind, 'sorted_in_folder'] = output_path

def get_convert_path(path):
    scene_id, scene_sub_id, img_id = path.split("_")
    convert_path = "{}_{}/color/{}".format(scene_id, scene_sub_id, img_id)
    return convert_path

def update_annotation_info():
    dst_folder = "output"
    mirror_paths = os.listdir(os.path.join(dst_folder, "mirror"))
    non_mirror_paths = os.listdir(os.path.join(dst_folder, "negative"))

    exclusion_paths = set(exception_paths)
    train_pos = set()
    train_neg = set()
    for path in mirror_paths:
        convert_path = get_convert_path(path)
        exclusion_paths.add(convert_path)
        train_pos.add(os.path.join(input_folder, convert_path))
    for path in non_mirror_paths:
        convert_path = get_convert_path(path)
        exclusion_paths.add(convert_path)
        train_neg.add(os.path.join(input_folder, convert_path))
    save_txt(os.path.join(dst_folder, "exclusion.txt"), exclusion_paths)
    
    annotated_paths = paths
    verify_list = [path for path in paths if path not in train_pos.union(train_neg)]
    save_txt(os.path.join(dst_folder,"verify.txt"), verify_list)

    old_train_pos = set(read_txt(os.path.join(train_val_info_folder, "train_positive.txt")))
    old_train_neg = set(read_txt(os.path.join(train_val_info_folder, "train_negative.txt")))
    train_pos |= old_train_pos
    train_neg |= old_train_neg
    save_txt(os.path.join(dst_folder,"train_positive.txt"), train_pos)
    save_txt(os.path.join(dst_folder,"train_negative.txt"), train_neg)

def make_folder(directory):
    """
    Make folder if it doesn't already exist
    :param directory: The folder destination path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def close_window():
    with open(output_file_names_path, 'w') as f:
        for line in output_file_names:
            f.write(line + '\n')
    root.destroy()

# The main bit of the script only gets exectured if it is directly called
if __name__ == "__main__":

    # Commenting out the initial input and puting input into preamble
    # Make input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='Input folder where the *tif images should be', required=True)
    parser.add_argument('-j', '--json_file_path', help='Json file consist of input file names', required=True)
    parser.add_argument('-l', '--labels', nargs='+', help='Possible labels in the images', required=True)
    parser.add_argument('-e', '--exclusion', help='Exclusion file', required=False)
    parser.add_argument('-o', '--output_file_path', help='Output file name', required=True)
    parser.add_argument('-i', '--train_val_info_folder', help='Train val info folder', required=True)
    parser.add_argument('--is_default', type=bool, default=True, help='mirror negative labels')
    args = parser.parse_args()

    # grab input arguments from args structure
    input_folder = args.folder
    labels = args.labels
    json_file_path = args.json_file_path
    output_file_names_path = args.output_file_path
    train_val_info_folder = args.train_val_info_folder
    is_default = args.is_default

    # Exception files
    exception_paths = []
    with open(args.exclusion, 'r') as e:
        for line in e:
            exception_paths.append(line)
    
    # Make folder for the new labels
    for label in args.labels:
        make_folder(os.path.join("output", label))

    # Read file names from json
    with open(json_file_path, 'r') as j:
        file_names = json.loads(j.read())

    # Sort by score
    file_names = sorted(file_names.items(), key=lambda d: d[1], reverse=True)

    new_file_names = []

    for name, score in file_names:
        if name not in exception_paths:
            new_file_names.append(name)
    paths = [input_folder + '/' + file_name for file_name in new_file_names]

    if copy_or_move == 'copy':
        try:
            df = pd.read_csv(df_path, header=0)
            # Store configuration file values
        except FileNotFoundError:
            df = pd.DataFrame(columns=["im_path", 'sorted_in_folder'])
            df.im_path = paths
            df.sorted_in_folder = paths
    if copy_or_move == 'move':
        df = pd.DataFrame(columns=["im_path", 'sorted_in_folder'])
        df.im_path = paths
        df.sorted_in_folder = paths

    if os.path.exists(output_file_names_path):
        with open(output_file_names_path, 'r') as o:
            for line in o:
                output_file_names.append(line.strip())

    
    # Start the GUI
    root = tk.Tk()
    root.protocol('WM_DELETE_WINDOW', close_window)
    app = ImageGui(root, labels, paths, is_default)
    root.mainloop()