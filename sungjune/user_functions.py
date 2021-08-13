"""
멀티캠퍼스 데이터사이언스 과정 - 팀프로젝트 2차
공공데이터를 활용한 머신러닝(딥러닝) 기반 예측 모델 구현 과정에서 사용할 유저 함수를 정의함
"""

import getpass
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

################################################################
# 사용자 이름 가져오기
################################################################

def get_f_path():
    # 사용자 이름 저장
    user_name = getpass.getuser()

    # 서버 이름이 lab10일 때 파일 위치
    if user_name == "lab10":
        folder_path = "raw_data/"

    # Local에서 파일 위치
    elif user_name == "sungjune":
        folder_path = "/Users/sungjune/Downloads/raw_data/"

    # 서버 이름이 나머지 일 때 파일 위치
    else:
        folder_path = "../lab10/raw_data/"

    return folder_path
################################################################


################################################################
# 이상치 / 결측치 확인
################################################################
def check_feature(datasets, feature):
    print("=" * 22, "describe", "=" * 22)
    try:
        print(datasets[feature].describe())
    except:
        print("can't describe. dtype is", datasets[feature].dtype)
    print("=" * 54)
    
    print("=" * 23, "unique", "=" * 23)
    print(datasets[feature].unique())
    print("=" * 54)

    print("=" * 20, "value_counts", "=" * 20)
    print(datasets[feature].value_counts())
    print("=" * 54)

    print("=" * 24, "isna", "=" * 24)
    print(datasets[feature].isna().value_counts())
    print("=" * 54)
    
    return None
################################################################


################################################################
# 그래프
################################################################

# 이상치 확인 - BOXPLOT
def box_plot(datasets, feature,  max_value=6):
    plt.figure(figsize=(15, 5))
    sns.set_theme(style="whitegrid")
    sns.boxplot(x=datasets[feature])

# bar 그래프를 그리는 함수
def bar_plot(feature):
    # label font size setting
    parameters = {"axes.labelsize": 16,
                "axes.titlesize": 16} # 안 됨
    plt.rcParams.update(parameters)

    # 1행 2열 개로 표 나눔
    fig, axes = plt.subplots(1, 2)

    # 표 크기 설정
    fig.set_size_inches(15, 7.5)

    # 테마 설정
    sns.set_theme(style="whitegrid")

    # 기본 데이터 설정
    temp1_ds = pre_train_data[["TARGET", feature]]
    # 글자 수 10개로 제한
    temp1_ds[feature] = temp1_ds[feature].str.slice(start=0, stop=10)
    temp2_ds = temp1_ds.loc[temp1_ds["TARGET"] == 1]

    # 전체 데이터 개수 표시
    sns.barplot(y=temp1_ds.iloc[:, 1].value_counts()[:max_value],
                x=temp1_ds.iloc[:, 1].value_counts()[:max_value].index,
                ax=axes[0]
                )
    axes[0].set(xlabel="Unique values",
                ylabel="Counts",
                title="Total"
            )
    axes[0].set_xticklabels(labels=temp1_ds.iloc[:, 1].value_counts()[:max_value].index,
                            rotation=25,
                            fontsize=13
                        )

    # TARGET == 1인 것만 표시
    sns.barplot(y=temp2_ds.iloc[:, 1].value_counts()[:max_value],
                x=temp2_ds.iloc[:, 1].value_counts()[:max_value].index,
                ax=axes[1]
            )
    axes[1].set(xlabel="Unique values",
                ylabel="Counts",
                title="Only DEFAULT",
            )
    axes[1].set_xticklabels(labels=temp2_ds.iloc[:, 1].value_counts()[:max_value].index,
                            rotation=25,
                            fontsize=13
                        )

    plt.show()

    return None

def bar_plot_pie(datasets, feature, max_value=6):
    # label font size setting
    parameters = {"axes.labelsize": 16,
                "axes.titlesize": 16} # 안 됨
    plt.rcParams.update(parameters)

    # 1행 2열 개로 표 나눔
    fig, axes = plt.subplots(1, 2)

    # 표 크기 설정
    fig.set_size_inches(15, 7.5)

    # 테마 설정
    sns.set_theme(style="whitegrid")

    # 기본 데이터 설정
    temp1_ds = datasets.loc[:, ["TARGET", feature]]
    # 글자 수 10개로 제한
    temp1_ds[feature] = temp1_ds[feature].str.slice(start=0, stop=10)
    temp2_ds = temp1_ds.loc[temp1_ds["TARGET"] == 1]

    # 전체 데이터 개수 표시
    sns.barplot(y=temp1_ds.iloc[:, 1].value_counts()[:max_value],
                x=temp1_ds.iloc[:, 1].value_counts()[:max_value].index,
                ax=axes[0]
                )
    axes[0].set(xlabel="Unique values",
                ylabel="Counts",
                title="Total"
            )
    axes[0].set_xticklabels(labels=temp1_ds.iloc[:, 1].value_counts()[:max_value].index,
                            rotation=25,
                            fontsize=13
                        )

    # TARGET == 1인 것만 표시
    pie = axes[1].pie(temp2_ds.iloc[:, 1].value_counts()[:max_value],
                      labels=temp2_ds.iloc[:, 1].value_counts()[:max_value].index,
                      autopct=lambda x: "{0:.1f}%".format(x)
                     )
    plt.show()

    return None
################################################################


################################################################
# 히트맵
################################################################

# 상관관계 히트맵
def corr_heatmap(datasets):
    corr = datasets.corr()
    plt.figure(figsize=(14, 14))
    sns.heatmap(corr, annot=True, fmt=".1g", vmin=-1, vmax=1, cmap="RdBu_r")


################################################################
# 스케일링 / 인코딩
################################################################

def scaling_and_encoding(datasets, scaling=True, encoding=True):
    # dtype이 object인 features 리스트 생성
    categorical_feature = \
        datasets.dtypes[datasets.dtypes == "object"].index.to_list()

    # dtype이 object가 아닌 numerical features 리스트 생성
    numerical_feature = \
        datasets.dtypes[datasets.dtypes != "object"].index.to_list()
    
    ##########################################################
    # 숫자형 features에 RobustScaler()적용
    ##########################################################
    if scaling == True:
        for feature in numerical_feature:
            rbscaler = RobustScaler()
            
            # RobustScaler로 datasets 변환
            # numpy.ndarray 반환됨
            temp_rbscaler = rbscaler.fit_transform(pd.DataFrame(datasets[feature]))

            # 데이터 수정
            datasets[feature] = temp_rbscaler
            # numpy.ndarray에서 pandas.DataFrame으로 변환할 경우
            # pd.DataFrame(ds_rbscaler, columns=temp_ds.columns)   
    ##########################################################

    ##########################################################
    # 문자형 features를 LabelEncoder()적용
    ##########################################################
    if encoding == True:
        for feature in categorical_feature:
            # LabelEncoder object 생성
            l_encod = LabelEncoder()

            # fit을 통해 인코딩 수행
            # numpy.ndarray 반환됨
            temp_encod = l_encod.fit_transform(datasets[feature].ravel())
            
            # 데이터 수정
            datasets[feature] = temp_encod

    return datasets


def test_scaler_set(datasets, feature):
    # RobustScaler object 생성
    rbscaler = RobustScaler()
    # RobustScaler로 datasets 변환
    temp_ds = datasets.drop(columns=[feature])
    ds_rbscaler = rbscaler.fit_transform(temp_ds)
    # numpy.ndarray 반환된 값 pandas.DataFrame으로 변환
    ds_rbscaler_df = pd.DataFrame(ds_rbscaler, columns=temp_ds.columns)

    # StandardScaler object 생성
    stscaler = StandardScaler()
    # StandardScaler로 datasets 변환
    temp_ds = datasets.drop(columns=[feature])
    ds_stscaler = stscaler.fit_transform(temp_ds)
    # numpy.ndarray 반환된 값 pandas.DataFrame으로 변환
    ds_stscaler_df = pd.DataFrame(ds_stscaler, columns=temp_ds.columns)

    # MinMaxScaler object 생성
    mmscaler = MinMaxScaler()
    # MinMaxScaler로 datasets 변환
    temp_ds = datasets.drop(columns=[feature])
    ds_mmscaler = mmscaler.fit_transform(temp_ds)
    # numpy.ndarray 반환된 값 pandas.DataFrame으로 변환
    ds_mmscaler_df = pd.DataFrame(ds_mmscaler, columns=temp_ds.columns)

    # 히스토그램 출력
    ds_rbscaler_df["AMT_DRAWINGS_CURRENT"].hist()
    ds_stscaler_df["AMT_DRAWINGS_CURRENT"].hist()
    ds_mmscaler_df["AMT_DRAWINGS_CURRENT"].hist()
################################################################


################################################################
# 기타함수
################################################################
def imput_other_f_mean(datasets, target_f, mean_f):
    """
    -------------------------------------
    |   |    target_f    |    mean_f    |
    -------------------------------------
    | 0 |       NA       |     100      |
    | 1 |       NA       |     200      |
    | 2 |        5       |     100      |
    | 3 |        7       |     100      |
    | 4 |       10       |     200      |
    -------------------------------------

    index 1 NA
    => mean_f == 100 and [index 2, index 3] is same value
    => index 1 NA = (5 + 7) / 2 = 6
    """
    # target feature가 na인 row index 반환
    na_index = datasets.loc[datasets[target_f].isna(), target_f].index

    # target feature 중 na인 값을
    # na인 행에서 mean feature값을 찾아, mean feature가 같은 값의 target feature 의평균(소수점 1의 자리에서 반올림)으로 대체함
    for i in na_index:
        # na인 행의 mean feature값
        target_mean_f_value = datasets.loc[i, mean_f]

        # 같은 값을 찾음
        same_value = datasets.loc[datasets[mean_f] == target_mean_f_value, target_f]
        # 같은 값을 찾지 못하면 가장 근사치를 찾음.
        if same_value.isnull:
            target_mean_f_value = \
                datasets.loc[datasets[mean_f] > target_mean_f_value, mean_f].min()
        # 같은 값을 찾음
        same_value = datasets.loc[datasets[mean_f] == target_mean_f_value, target_f]

        # 값을 대체함
        datasets.loc[i, target_f] = \
            round(datasets.loc[datasets[mean_f] == target_mean_f_value, target_f].mean(), 1)
            # mean_f가 target_mean_f_value인 값을 찾는다.
            # -> 그 행에서 target_f값을 찾아서 평균을 취한다.
    return datasets[target_f]


# 결측치 확인
def check_missing_value(datasets):
    ################################################
    # 출력 값을 맞추기 위해 설정하는 부분
    ################################################
    max_1 = datasets.columns.str.len().max()
    # 전체 DataFrame에서 object가 아닌 값 중 max값을 반환함
    max_2_list = []
    for feat in datasets.columns:
        try:
            if datasets[feat].max().dtype != object:
                max_2_list.append(datasets[feat].max())
        except:
            pass
    max_2 = len(format(int(max(max_2_list)), ","))
    ################################################

    no_na = True
    for i, f in enumerate(datasets.columns):
        # 결측값이 있는 feature만 출력
        if datasets[f].isna().sum() != 0:
            no_na = Fasle
            print(format(i, "3d"), 
                format(f, "^%ds" %(max_1 + 1)), 
                format(datasets[f].isna().sum(), ">%d,d" %(max_2 + 1))
                )
    if no_na:
        print("결측치 없음!")
################################################################


# matplot 한글 출력
def korean():
    import platform
    # 운영체제별 한글 폰트 설정
    if platform.system() == 'Darwin': # Mac 환경 폰트 설정
        plt.rc('font', family='AppleGothic')
    elif platform.system() == 'Windows': # Windows 환경 폰트 설정
        plt.rc('font', family='Malgun Gothic')

    plt.rc('axes', unicode_minus=False) # 마이너스 폰트 설정

    # 글씨 선명하게 출력하는 설정
    %config InlineBackend.figure_format = 'retina'