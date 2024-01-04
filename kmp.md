kmp

knext

next

nextval

```
#include<bits/stdc++.h>
using namespace std;
#define ll long long
const int N=2e6+10;

string s1,s2;
int len1,len2;
int k;
int knxt[N];//next数组（不要写next，在CF上是关键字，会CE）
int nxt[N];
void kmp(string s){//我懒得讲KMP了怎么办啊QwQ
    int l=s.size();
    s=' '+s;//前面加个空格，让s下标从1开始
    knxt[0]=knxt[1]=0;//初始化next数组
    int j=0;//下标
    for(int i=2;i<=l;i++){
        while(j>0&&s[i]!=s[j+1])j=knxt[j];//如果不匹配，就借用已知信息跳到next
        if(s[i]==s[j+1])j++;//匹配成功！j++;
        knxt[i]=j;//更新next数组的信息
    } 
    for(int i=1;i<=l;i++)
    {
        cout<<knxt[i]<<" ";
    }
    nxt[1]=0;
    for(int i=2;i<=l;i++)
    {
        nxt[i]=knxt[i-1]+1;
    }
    cout<<endl;
    for(int i=1;i<=l;i++)
    {
        cout<<nxt[i]<<" ";
    }
}

int nxtval[N];
void getval(string s)
{
    s=' '+s;
    nxtval[0]=0;
    for(int i=1;i<=s.size();i++)
    {
        if(s[i]==s[nxt[i]])
            nxtval[i]=nxtval[nxt[i]];
        else
            nxtval[i]=nxt[i];
    }
    cout<<endl;
    for(int i=1;i<s.size();i++)
    {
        cout<<nxtval[i]<<" ";
    }
}
void solve()
{
    cin>>s1;
    kmp(s1);
    getval(s1);

    return ;
}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);cout.tie(0);
    solve();    
    return 0;
} 
```

kmpval

```
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const int INF = 0x3f3f3f3f;
const int N=2e6+10;
int nextval[N];
void kmpval(string s)
{
	int i=0;int j=1;
	int n=s.size();
	nextval[0]=nextval[1]=0;
	while(j<n)
	{
		if(i==0 or s[i]==s[j])
		{
			if(s[i+1]==s[j+1])
			{
				nextval[j+1]=nextval[i+1];
			}
			else
			{
				nextval[j+1]=i+1;
			}
			i++;j++;
		}
		else
		{
			i=nextval[i];
		}

	}
	for(int i=0;i<n;i++)
	{
		cout<<nextval[i]<<" ";
	}
}
int main()
{
    string s;
    cin>>s;
    kmpval(s);
    
    return 0;
} 

```

