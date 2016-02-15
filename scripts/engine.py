import web
import parse_forest
from build_supervised_tree import TreeNode
from build_supervised_tree import LabelCount


p_forest = parse_forest.ParseForest()

urls = ('/', 'Index')
render = web.template.render('templates',)


class Index:
    # return the initial web page
    def GET(self):
        web.header("Content-Type", "text/html; charset=utf-8")
        return render.index('', '', '')

    # return the rendered web page after the file is uploaded
    def POST(self):
        print("Processing image...")
        x = web.input(myfile={})
        # change this to the directory you want to store the file in.
        filedir = 'static/uploads'
        if 'myfile' in x: # to check if the file-object is created
            # replaces the windows-style slashes with linux ones.
            filepath = x.myfile.filename.replace('\\','/')
            # splits the and chooses the last part (the filename with extension)
            filename = filepath.split('/')[-1]
            # creates the file where the uploaded file should be stored
            fout = open(filedir +'/'+ filename,'wb')
            # writes the uploaded file to the newly created file.
            fout.write(x.myfile.file.read())
            # closes the file, upload complete.
            fout.close()

            outfile = filedir +'/'+filename
            s_rc, s_rc = p_forest.im2res(outfile)
            print s_rc
            print s_rc

        print("Done.")
        return render.index(outfile, s_rc, s_rc)


if __name__ == "__main__":
    app = web.application(urls, globals(), autoreload=False)
    app.run()


# from scipy.io import loadmat
# train_path = utilites.get_file_list(utilites.getAbsPath('static/Corel5K/corel5k_train_list.txt'))
# utilites.saveVariableToFile(train_path, "static/Corel5K/train_file_path.pkl")
