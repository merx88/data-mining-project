import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from datetime import datetime, timedelta
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class NetflixSubscriptionAnalyzer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self._preprocess_data()
    
    def _preprocess_data(self):
        """데이터 전처리"""
        try:
            self.df['Join_Date'] = pd.to_datetime(self.df['Join Date'], format='%d-%m-%y')
            self.df['Last_Payment_Date'] = pd.to_datetime(self.df['Last Payment Date'], format='%d-%m-%y')

            self.df['Subscription_Length'] = ((self.df['Last_Payment_Date'] - self.df['Join_Date']).dt.days / 30).round()
            
            recent_payment_threshold = 14  # 2주
            last_payment_gap = (self.df['Last_Payment_Date'].max() - self.df['Last_Payment_Date']).dt.days
            
            short_subscription = self.df['Subscription_Length'] < 3
            
            low_revenue = self.df['Monthly Revenue'] < self.df['Monthly Revenue'].mean()
            
            self.df['Churn_Risk'] = (
                (last_payment_gap > recent_payment_threshold).astype(int) * 0.4 +  
                short_subscription.astype(int) * 0.3 +                            
                low_revenue.astype(int) * 0.3                                
            )
            
            self.df['Monthly_Revenue'] = pd.to_numeric(self.df['Monthly Revenue'])
            
            self.user_features = pd.get_dummies(self.df[['Subscription Type', 'Country', 'Device', 'Gender']])
        except Exception as e:
            print(f"데이터 전처리 중 오류 발생: {str(e)}")

    def generate_association_rules(self):
        """연관 규칙 생성"""
        try:
            features = ['Subscription Type', 'Device', 'Country', 'Churn_Risk']
            df_encoded = pd.get_dummies(self.df[features])
            
            frequent_itemsets = apriori(df_encoded, 
                                      min_support=0.1, 
                                      use_colnames=True)
            
            if len(frequent_itemsets) > 0:
                rules = association_rules(frequent_itemsets, 
                                        metric="confidence",
                                        min_threshold=0.5)
                return rules.sort_values('lift', ascending=False)
            return pd.DataFrame()
            
        except Exception as e:
            print(f"연관 규칙 생성 중 오류: {str(e)}")
            return pd.DataFrame()

    def get_insights(self):
        """주요 인사이트 도출"""
        try:
            insights = {
                '구독유형별_이탈위험': self.df.groupby('Subscription Type')['Churn_Risk'].mean().round(3),
                '국가별_이탈위험': self.df.groupby('Country')['Churn_Risk'].mean().round(3),
                '디바이스별_이탈위험': self.df.groupby('Device')['Churn_Risk'].mean().round(3),
                '연령대별_이탈위험': self.df.groupby(
                    pd.cut(self.df['Age'], 
                          bins=[0,25,35,45,100],
                          labels=['25세 이하','26-35세','36-45세','46세 이상']))['Churn_Risk'].mean().round(3)
            }
            
            insights['평균_구독기간'] = round(self.df['Subscription_Length'].mean(), 1)
            insights['평균_월수익'] = round(self.df['Monthly_Revenue'].mean(), 1)
            
            insights['이탈_위험_분포'] = {
                '높음 (0.7-1.0)': len(self.df[self.df['Churn_Risk'] >= 0.7]),
                '중간 (0.4-0.7)': len(self.df[(self.df['Churn_Risk'] >= 0.4) & (self.df['Churn_Risk'] < 0.7)]),
                '낮음 (0-0.4)': len(self.df[self.df['Churn_Risk'] < 0.4])
            }
            
            return insights
            
        except Exception as e:
            print(f"인사이트 도출 중 오류: {str(e)}")
            return {}

    def get_target_strategies(self):
        """타겟 마케팅 전략 도출"""
        try:
            insights = {
                '구독유형별_이탈위험': self.df.groupby('Subscription Type')['Churn_Risk'].mean(),
                '국가별_이탈위험': self.df.groupby('Country')['Churn_Risk'].mean(),
                '디바이스별_이탈위험': self.df.groupby('Device')['Churn_Risk'].mean(),
                '연령대별_이탈위험': self.df.groupby(pd.cut(self.df['Age'], 
                                                bins=[0,25,35,45,100],
                                                labels=['25세 이하','26-35세','36-45세','46세 이상']))['Churn_Risk'].mean()
            }
            
            high_risk_segments = {
                'subscription': insights['구독유형별_이탈위험'].nlargest(1).index[0],
                'country': insights['국가별_이탈위험'].nlargest(1).index[0],
                'device': insights['디바이스별_이탈위험'].nlargest(1).index[0],
                'age_group': insights['연령대별_이탈위험'].nlargest(1).index[0]
            }
            return high_risk_segments
            
        except Exception as e:
            print(f"타겟 전략 도출 중 오류 발생: {str(e)}")
            return {
                'subscription': 'Basic',
                'country': 'United States',
                'device': 'Smartphone',
                'age_group': '26-35세'
            }

    def suggest_retention_programs(self):
        """구독 유지 프로그램 제안"""
        try:
            churn_by_type = self.df.groupby('Subscription Type')['Churn_Risk'].mean()
            churn_by_device = self.df.groupby('Device')['Churn_Risk'].mean()
            
            high_risk_type = churn_by_type.idxmax()
            high_risk_device = churn_by_device.idxmax()
            
            return {
                'high_risk_users': self.df[self.df['Churn_Risk']]['User ID'].tolist(),
                'recommended_actions': [
                    f"타겟 세그먼트 ({high_risk_type}) 맞춤형 할인",
                    f"디바이스 ({high_risk_device}) 최적화된 서비스 개선",
                    "장기 구독자 특별 혜택 프로그램"
                ],
                'churn_rates': {
                    'subscription': churn_by_type.to_dict(),
                    'device': churn_by_device.to_dict()
                }
            }
            
        except Exception as e:
            print(f"구독 유지 프로그램 제안 중 오류: {str(e)}")
            return {}

    def perform_clustering(self):
        features = ['Age', 'Monthly_Revenue', 'Subscription_Length']
        X = self.df[features]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=4, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        return self.df.groupby('Cluster')[features + ['Churn_Risk']].mean()

def main():
    try:
        print("\n=== 데이터 로딩 및 전처리 ===")
        analyzer = NetflixSubscriptionAnalyzer('./Netflix Userbase.csv')
        
        print("\n=== 주요 인사이트 분석 ===")
        insights = analyzer.get_insights()
        if insights:
            for key, value in insights.items():
                if key != '상위_연관규칙':
                    print(f"\n{key}:")
                    print(value)
        
        print("\n=== 타겟 마케팅 전략 도출 ===")
        strategies = analyzer.get_target_strategies()
        if strategies:
            for key, value in strategies.items():
                print(f"{key}: {value}")
        
        print("\n=== 구독 유지 프로그램 제안 ===")
        retention = analyzer.suggest_retention_programs()
        if retention:
            print("\n추천 액션:")
            for action in retention['recommended_actions']:
                print(f"- {action}")
            
            if 'churn_rates' in retention:
                print("\n이탈률 분석:")
                print("구독 유형별:", retention['churn_rates']['subscription'])
                print("디바이스별:", retention['churn_rates']['device'])
                
    except Exception as e:
        print(f"예상치 못한 오류 발생: {str(e)}")
        import traceback
        print("상세 에러 메시지:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
