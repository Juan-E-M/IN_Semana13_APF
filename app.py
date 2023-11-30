from flask import Flask, render_template, jsonify, request
from IntelNeg import neighbors, binary_to_pandas_with_stats, computeManhattanDistance, movie_lens_to_binary
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

movie_lens_to_binary('ratings.dat', 'ratings.bin')
consolidate = binary_to_pandas_with_stats('ratings.bin')
def extract_user_data_from_dat(dat_file, user_id):
    ratings = pd.read_csv(dat_file, sep='\t', header=None,
                          names=['userId', 'movieId', 'rating', 'rating_timestamp'])
    user_data = ratings[ratings['userId'] == user_id]
    consolidated_user_df = user_data.groupby(['userId', 'movieId'])['rating'].mean().unstack()
    return consolidated_user_df

def merge_dataframes_with_stats_and_user(bin_df, user_id):
    merged_df = pd.concat([bin_df, user_df], axis=1)
    return merged_df


def check_user_in_dataframe(df, user_id):
    return user_id in df.index

@app.route("/", methods=['POST','GET'])
def hello():
    
    if request.method == 'POST':
        user1 = int(request.form['user1'])
        vecinos = neighbors(user1, consolidate, computeManhattanDistance)
        response_data = {'data': vecinos}
        return jsonify(response_data)

    return render_template('index.html')


@app.route('/data', methods=['GET'])
def get_data():
    user_id = int(request.args.get('user_id'))
    check = check_user_in_dataframe(consolidate,user_id)
    if check:
        user_df = extract_user_data_from_dat('ratings.dat',user_id)
        consolidate2 = merge_dataframes_with_stats_and_user(consolidate, user_id)
    else:
        consolidate2 = consolidate
    vecinos = neighbors(user1, consolidate, computeManhattanDistance)
    response_data = {'data': vecinos}
    vecinos = computeNearestNeighbor(user_id, consolidate2, computeManhattanDistance):
    return jsonify(vecinos)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)