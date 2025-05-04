import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import helpers


class DataPreprocessing:
    def __init__(self, dataframe):
        """
        Initialize with a combined DataFrame (train + test).

        Args:
            dataframe (pd.DataFrame): Combined DataFrame with train (ürün fiyatı non-null) and test (ürün fiyatı null) data.
        """
        self.df = dataframe.copy()

    def preprocess(self, is_test_only=False):
        """
        Preprocess the data and return train/validation splits or test data.

        Args:
            is_test_only (bool): If True, return test data and ids; otherwise, return train/validation splits.

        Returns:
            If is_test_only=True: (test_data, test_ids)
            If is_test_only=False: (X_train, X_val, y_train, y_val)
        """
        self.handle_outliers()
        self.handle_missing_values()
        self.feature_engineering()
        self.drop_unnecessary_columns()
        self.encode_features()

        if is_test_only:
            # Test verisini al ve 'id'yi düşürmeden önce sakla
            test_data = self.df[self.df['ürün fiyatı'].isnull()].drop('ürün fiyatı', axis=1)
            test_ids = test_data["id"].copy()  # 'id'yi sakla
            test_data = test_data.drop(columns=['id'])  # 'id'yi test verisinden çıkar
            return test_data, test_ids  # 'test_ids' ile birlikte döndür
        else:
            # Eğitim verisini al ve 'id'yi düşür
            train_data = self.df[self.df['ürün fiyatı'].notnull()]
            train_data = train_data.drop(columns=['id'])  # 'id'yi düşür

            X = train_data.drop('ürün fiyatı', axis=1)
            y = train_data['ürün fiyatı']
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_val, y_train, y_val

    def handle_outliers(self):
        """
        Handle outliers using IQR method for numerical columns (excluding ürün fiyatı).
        """
        num_cols = self.df.select_dtypes(include=np.number).columns
        num_cols = [col for col in num_cols if col != 'ürün fiyatı']  # Hedef değişkeni hariç tut
        for col in num_cols:
            if helpers.check_outlier(self.df, col):
                self.df = helpers.replace_with_thresholds(self.df, col)

    def handle_missing_values(self):
        """
        Handle missing values: mean for numerical (excluding ürün fiyatı), mode for categorical.
        """
        num_cols = self.df.select_dtypes(include=np.number).columns
        num_cols = [col for col in num_cols if col != 'ürün fiyatı']
        self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].mean())
        cat_cols = self.df.select_dtypes(include='object').columns
        for col in cat_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

    def feature_engineering(self):
        """
        Create new features for the dataset.
        """
        # Besin değeri ile ilgili özellikler
        self.df['besin_değeri_log'] = np.log1p(self.df['ürün besin değeri'])  # Log dönüşümü

        # Ürün kategorisi bazlı ortalama besin değeri
        self.df['kategori_ortalama_besin'] = self.df.groupby('ürün kategorisi')['ürün besin değeri'].transform('mean')

    def drop_unnecessary_columns(self):
        """
        Drop unnecessary columns.
        """
        columns_to_drop = ['ürün üretim yeri', 'market', 'şehir']  # Tek değerli sütunlar
        self.df.drop(columns=[col for col in columns_to_drop if col in self.df.columns], inplace=True)

    def encode_features(self):
        """
        Encode categorical features (ürün, ürün kategorisi).
        """
        cat_cols, cat_but_car, num_cols = helpers.grab_col_names(self.df)

        # Binary veya düşük kardinaliteli sütunlar için label encoding
        binary_cols = [col for col in cat_cols if self.df[col].nunique() <= 3]  # Örneğin, ürün kategorisi
        for col in binary_cols:
            self.df = helpers.label_encoder(self.df, col)

        # Yüksek kardinaliteli sütunlar (örneğin, ürün) için target encoding
        high_cardinality_cols = cat_but_car + [col for col in cat_cols if col not in binary_cols]
        for col in high_cardinality_cols:
            if col in self.df.columns:
                # Train verisi için hedef ortalaması hesapla
                train_data = self.df[self.df['ürün fiyatı'].notnull()]
                target_means = train_data.groupby(col)['ürün fiyatı'].mean()
                # Tüm veriye ortalamaları uygula, bilinmeyen değerler için genel ortalama
                self.df[col] = self.df[col].map(target_means).fillna(train_data['ürün fiyatı'].mean())

        # Kalan kategorik sütunlar için one-hot encoding
        remaining_cat_cols = [col for col in cat_cols if col not in binary_cols and col not in high_cardinality_cols]
        if remaining_cat_cols:
            self.df = helpers.one_hot_encoder(self.df, remaining_cat_cols, drop_first=True)

        # Hala kategorik sütun kalmışsa hata fırlat
        remaining_object_cols = self.df.select_dtypes(include='object').columns.tolist()
        if remaining_object_cols:
            raise ValueError(f"Categorical columns not fully encoded: {remaining_object_cols}")