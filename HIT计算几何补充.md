# HIT计算几何补充

[TOC]

## 0.极角序

　第三种方法按象限从小到大排序 再按极角从小到大排序是在有特殊需求的时候才会用到，这里不做比较。

　　关于第一种方法，利用atan2排序，他和利用叉积排序的主要区别在精度和时间上。

　　具体对比：时间：相较于计算叉积，利用atan2时间快，这个时间会快一点（记得做过一个题用atan2排序过了，用叉积的T了）

　　　　　　  精度： atan2精度不如叉积高，做过一个题用anat2因为精度问题WA了。

　　所以两种方法根据情况选择一种合适的使用。

### 1.$atan2(double$  $ y,$$double$  $x)$单位是弧度,范围$( − π , π ] $

#### 函数返回的是原点至点$ (x,y)$的方位角，即与$x$轴的夹角

```cpp
bool cmp(Point a, Point b) {
    if(dcmp(atan2(a.y, a.x) - atan2(b.y, b.x)) == 0) //dcmp为判断浮点数是否为0的函数
        return a.x < b.x;
    return atan2(a.y, a.x) < atan2(b.y, b.x);
}
```



### 2.利用向量叉乘排序

#### 

```
struct point//存储点
{
    double x,y;
};

double cross(double x1,double y1,double x2,double y2)　//计算叉积
{
    return (x1*y2-x2*y1);
}

double compare(point a,point b,point c)//计算极角
{
    return cross((b.x-a.x),(b.y-a.y),(c.x-a.x),(c.y-a.y));
}

bool cmp2(point a,point b) 
{
    point c;//原点
    c.x = 0;
    c.y = 0;
    if(compare(c,a,b)==0)//计算叉积，函数在上面有介绍，如果叉积相等，按照X从小到大排序
        return a.x<b.x;
    else return compare(c,a,b)>0;
}
```

#### 

### 3.先按象限从小到大排序 再按极角从小到大排序

```
int Quadrant(point a)　　//象限排序，注意包含四个坐标轴
{
    if(a.x>0&&a.y>=0)  return 1;
    if(a.x<=0&&a.y>0)  return 2;
    if(a.x<0&&a.y<=0)  return 3;
    if(a.x>=0&&a.y<0)  return 4;
}


bool cmp3(point a,point b)  //先按象限从小到大排序 再按极角从小到大排序
{
    if(Quadrant(a)==Quadrant(b))//返回值就是象限
        return cmp1(a,b);
    else Quadrant(a)<Quadrant(b);
}
```



## 1.三角形

### 三角形重心

```cpp
struct Point {
    double x, y;
};
struct Line{
    Point a, b;
};
Point Intersection(Line u, Line v){
	Point ret = u.a;
	double t1 = (u.a.x - v.a.x)*(v.a.y - v.b.y) - (u.a.y - v.a.y)*(v.a.x - v.b.x);
	double t2 =	(u.a.x - u.b.x)*(v.a.y - v.b.y) - (u.a.y - u.b.y)*(v.a.x - v.b.x);
	double t = t1/t2;
	ret.x += (u.b.x - u.a.x)*t;
	ret.y += (u.b.y - u.a.y)*t;
	return ret;
}
//三角形重心
//到三角形三顶点距离的平方和最小的点
//三角形内到三边距离之积最大的点
Point barycenter(Point a, Point b, Point c){
	Line u, v;
	u.a.x = (a.x + b.x)/2;
	u.a.y = (a.y + b.y)/2;
	u.b = c;
	v.a.x = (a.x + c.x)/2;
	v.a.y = (a.y + c.y)/2;
	v.b = b;
	return Intersection(u, v);
}
```

### 三角形费马点

```cpp
struct Point {
    double x, y;
};
struct Line{
    Point a, b;
};
inline double Dist(Point p1, Point p2){
	return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}
//三角形费马点
//到三角形三顶点距离之和最小的点
Point Ferment(Point a, Point b, Point c){
	Point u, v;
	double step = fabs(a.x) + fabs(a.y) + fabs(b.x) + fabs(b.y) + fabs(c.x) + fabs(c.y);
	u.x = (a.x + b.x + c.x)/3;
	u.y = (a.y + b.y + c.y)/3;
	while(step >1e-10){
        for(int k = 0; k < 10; step /= 2, k++){
            for(int i = -1; i <= 1; ++i){
                for(int j = -1; j <= 1; ++j){
					v.x = u.x + step*i;
					v.y = u.y + step*j;
					double t1 = Dist(u, a) + Dist(u, b) + Dist(u, c);
					double t2 = Dist(v, a) + Dist(v, b) + Dist(v, c);
					if (t1 > t2) u = v;
				}
			}
		}
	}
	return u;
}
```

## 2.网格图

### 多边形与网格点

```cpp
struct Point{
    int x, y;
};
//多边形上的网格点个数
int Onedge(int n, Point* p){
	int ret = 0;
	for(int i = 0; i < n; ++i)
        ret += __gcd(abs(p[i].x - p[(i + 1)%n].x), abs(p[i].y - p[(i + 1)%n].y));
	return ret;
}
//多边形内的网格点个数
int Inside(int n, Point* p){
	int ret = 0;
	for (int i = 0; i < n; ++i)
		ret += p[(i + 1)%n].y*(p[i].x - p[(i + 2)%n].x);
    ret = (abs(ret) - Onedge(n, p))/2 + 1;
	return ret;
}
```

## 3.点与线

### 手动实现

```cpp
struct Point{
    double x, y;
    Point(double x = 0, double y = 0):x(x),y(y){}
};
typedef Point Vector;
Vector operator + (Vector A, Vector B){
    return Vector(A.x+B.x, A.y+B.y);
}
Vector operator - (Point A, Point B){
    return Vector(A.x-B.x, A.y-B.y);
}
Vector operator * (Vector A, double p){
    return Vector(A.x*p, A.y*p);
}
Vector operator / (Vector A, double p){
    return Vector(A.x/p, A.y/p);
}
bool operator < (const Point& a, const Point& b){
    if(a.x == b.x)
        return a.y < b.y;
    return a.x < b.x;
}
const double eps = 1e-6;
int sgn(double x){
    if(fabs(x) < eps)
        return 0;
    if(x < 0)
        return -1;
    return 1;
}
bool operator == (const Point& a, const Point& b){
    if(sgn(a.x-b.x) == 0 && sgn(a.y-b.y) == 0)
        return true;
    return false;
}
double Dot(Vector A, Vector B){
    return A.x*B.x + A.y*B.y;
}
double Length(Vector A){
    return sqrt(Dot(A, A));
}
double Angle(Vector A, Vector B){
    return acos(Dot(A, B)/Length(A)/Length(B));
}
double Cross(Vector A, Vector B){
    return A.x*B.y-A.y*B.x;
}
double Area2(Point A, Point B, Point C){
    return Cross(B-A, C-A);
}
Vector Rotate(Vector A, double rad){//rad为弧度 且为逆时针旋转的角
    return Vector(A.x*cos(rad)-A.y*sin(rad), A.x*sin(rad)+A.y*cos(rad));
}
Vector Normal(Vector A){//向量A左转90°的单位法向量
    double L = Length(A);
    return Vector(-A.y/L, A.x/L);
}
bool ToLeftTest(Point a, Point b, Point c){
    return Cross(b - a, c - b) > 0;
}
```

### 直线定义

```cpp
struct Line{//直线定义
    Point v, p;
    Line(Point v, Point p):v(v), p(p) {}
    Point point(double t){//返回点P = v + (p - v)*t
        return v + (p - v)*t;
    }
};
```

### 求两直线交点

```cpp
//调用前需保证 Cross(v, w) != 0
Point GetLineIntersection(Point P, Vector v, Point Q, Vector w){
    Vector u = P-Q;
    double t = Cross(w, u)/Cross(v, w);
    return P+v*t;
}
```

###  求点在直线上的投影点

```cpp
//点P在直线AB上的投影点
Point GetLineProjection(Point P, Point A, Point B){
    Vector v = B-A;
    return A+v*(Dot(v, P-A)/Dot(v, v));
}
```

###  求点到线段距离

```cpp
//点P到线段AB距离公式
double DistanceToSegment(Point P, Point A, Point B){
    if(A == B)
        return Length(P-A);
    Vector v1 = B-A, v2 = P-A, v3 = P-B;
    if(dcmp(Dot(v1, v2)) < 0)
        return Length(v2);
    if(dcmp(Dot(v1, v3)) > 0)
        return Length(v3);
    return DistanceToLine(P, A, B);
}
```

### 求点到直线距离

```cpp
//点P到直线AB距离
double DistanceToLine(Point P, Point A, Point B){
    Vector v1 = B-A, v2 = P-A;
    return fabs(Cross(v1, v2)/Length(v1));
}//不去绝对值，得到的是有向距离
```

## 4.矢量运算all

```cpp
#include <iostream>
#include <cmath> 
#include <vector> 
#include <algorithm> 
#define MAX_N 100
using namespace std; 
 
 
///
//常量区
const double INF        = 1e10;     // 无穷大 
const double EPS        = 1e-15;    // 计算精度 
const int LEFT          = 0;        // 点在直线左边 
const int RIGHT         = 1;        // 点在直线右边 
const int ONLINE        = 2;        // 点在直线上 
const int CROSS         = 0;        // 两直线相交 
const int COLINE        = 1;        // 两直线共线 
const int PARALLEL      = 2;        // 两直线平行 
const int NOTCOPLANAR   = 3;        // 两直线不共面 
const int INSIDE        = 1;        // 点在图形内部 
const int OUTSIDE       = 2;        // 点在图形外部 
const int BORDER        = 3;        // 点在图形边界 
const int BAOHAN        = 1;        // 大圆包含小圆
const int NEIQIE        = 2;        // 内切
const int XIANJIAO      = 3;        // 相交
const int WAIQIE        = 4;        // 外切
const int XIANLI        = 5;        // 相离
const double pi		   = acos(-1.0)  //圆周率
/// 
 
 
///
//类型定义区
struct Point {              // 二维点或矢量 
    double x, y; 
    double angle, dis; 
    Point() {} 
    Point(double x0, double y0): x(x0), y(y0) {} 
}; 
struct Point3D {            //三维点或矢量 
    double x, y, z; 
    Point3D() {} 
    Point3D(double x0, double y0, double z0): x(x0), y(y0), z(z0) {} 
}; 
struct Line {               // 二维的直线或线段 
    Point p1, p2; 
    Line() {} 
    Line(Point p10, Point p20): p1(p10), p2(p20) {} 
}; 
struct Line3D {             // 三维的直线或线段 
    Point3D p1, p2; 
    Line3D() {} 
    Line3D(Point3D p10, Point3D p20): p1(p10), p2(p20) {} 
}; 
struct Rect {              // 用长宽表示矩形的方法 w, h分别表示宽度和高度 
    double w, h; 
 Rect() {}
 Rect(double _w,double _h) : w(_w),h(_h) {}
}; 
struct Rect_2 {             // 表示矩形，左下角坐标是(xl, yl)，右上角坐标是(xh, yh) 
    double xl, yl, xh, yh; 
 Rect_2() {}
 Rect_2(double _xl,double _yl,double _xh,double _yh) : xl(_xl),yl(_yl),xh(_xh),yh(_yh) {}
}; 
struct Circle {            //圆
 Point c;
 double r;
 Circle() {}
 Circle(Point _c,double _r) :c(_c),r(_r) {}
};
typedef vector<Point> Polygon;      // 二维多边形     
typedef vector<Point> Points;       // 二维点集 
typedef vector<Point3D> Points3D;   // 三维点集 
/// 
 
 
///
//基本函数区
inline double max(double x,double y) 
{ 
    return x > y ? x : y; 
} 
inline double min(double x, double y) 
{ 
    return x > y ? y : x; 
} 
inline bool ZERO(double x)              // x == 0 
{ 
    return (fabs(x) < EPS); 
} 
inline bool ZERO(Point p)               // p == 0 
{ 
    return (ZERO(p.x) && ZERO(p.y)); 
} 
inline bool ZERO(Point3D p)              // p == 0 
{ 
    return (ZERO(p.x) && ZERO(p.y) && ZERO(p.z)); 
} 
inline bool EQ(double x, double y)      // eqaul, x == y 
{ 
    return (fabs(x - y) < EPS); 
} 
inline bool NEQ(double x, double y)     // not equal, x != y 
{ 
    return (fabs(x - y) >= EPS); 
} 
inline bool LT(double x, double y)     // less than, x < y 
{ 
    return ( NEQ(x, y) && (x < y) ); 
} 
inline bool GT(double x, double y)     // greater than, x > y 
{ 
    return ( NEQ(x, y) && (x > y) ); 
} 
inline bool LEQ(double x, double y)     // less equal, x <= y 
{ 
    return ( EQ(x, y) || (x < y) ); 
} 
inline bool GEQ(double x, double y)     // greater equal, x >= y 
{ 
    return ( EQ(x, y) || (x > y) ); 
} 
// 注意！！！ 
// 如果是一个很小的负的浮点数 
// 保留有效位数输出的时候会出现-0.000这样的形式， 
// 前面多了一个负号 
// 这就会导致错误！！！！！！ 
// 因此在输出浮点数之前，一定要调用次函数进行修正！ 
inline double FIX(double x) 
{ 
    return (fabs(x) < EPS) ? 0 : x; 
} 
// 
 
 
/
//二维矢量运算 
bool operator==(Point p1, Point p2)  
{ 
    return ( EQ(p1.x, p2.x) &&  EQ(p1.y, p2.y) ); 
} 
bool operator!=(Point p1, Point p2)  
{ 
    return ( NEQ(p1.x, p2.x) ||  NEQ(p1.y, p2.y) ); 
} 
bool operator<(Point p1, Point p2) 
{ 
    if (NEQ(p1.x, p2.x)) { 
        return (p1.x < p2.x); 
    } else { 
        return (p1.y < p2.y); 
    } 
} 
Point operator+(Point p1, Point p2)  
{ 
    return Point(p1.x + p2.x, p1.y + p2.y); 
} 
Point operator-(Point p1, Point p2)  
{ 
    return Point(p1.x - p2.x, p1.y - p2.y); 
} 
double operator*(Point p1, Point p2) // 计算叉乘 p1 × p2 
{ 
    return (p1.x * p2.y - p2.x * p1.y); 
} 
double operator&(Point p1, Point p2) { // 计算点积 p1·p2 
    return (p1.x * p2.x + p1.y * p2.y); 
} 
double Norm(Point p) // 计算矢量p的模 
{ 
    return sqrt(p.x * p.x + p.y * p.y); 
} 
// 把矢量p旋转角度angle (弧度表示) 
// angle > 0表示逆时针旋转 
// angle < 0表示顺时针旋转 
Point Rotate(Point p, double angle) 
{ 
    Point result; 
    result.x = p.x * cos(angle) - p.y * sin(angle); 
    result.y = p.x * sin(angle) + p.y * cos(angle); 
    return result; 
} 
// 
 
 
// 
//三维矢量运算 
bool operator==(Point3D p1, Point3D p2)  
{ 
    return ( EQ(p1.x, p2.x) && EQ(p1.y, p2.y) && EQ(p1.z, p2.z) ); 
} 
bool operator<(Point3D p1, Point3D p2) 
{ 
    if (NEQ(p1.x, p2.x)) { 
        return (p1.x < p2.x); 
    } else if (NEQ(p1.y, p2.y)) { 
        return (p1.y < p2.y); 
    } else { 
        return (p1.z < p2.z); 
    } 
} 
Point3D operator+(Point3D p1, Point3D p2)  
{ 
    return Point3D(p1.x + p2.x, p1.y + p2.y, p1.z + p2.z); 
} 
Point3D operator-(Point3D p1, Point3D p2)  
{ 
    return Point3D(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z); 
} 
Point3D operator*(Point3D p1, Point3D p2) // 计算叉乘 p1 x p2 
{ 
    return Point3D(p1.y * p2.z - p1.z * p2.y, 
        p1.z * p2.x - p1.x * p2.z, 
        p1.x * p2.y - p1.y * p2.x );         
} 
double operator&(Point3D p1, Point3D p2) { // 计算点积 p1·p2 
    return (p1.x * p2.x + p1.y * p2.y + p1.z * p2.z); 
} 
double Norm(Point3D p) // 计算矢量p的模 
{ 
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z); 
} 
 
 
// 
 
 
/
//点.线段.直线问题
//
double Distance(Point p1, Point p2) //2点间的距离
{
 return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
}
double Distance(Point3D p1, Point3D p2) //2点间的距离,三维
{
 return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
}
double Distance(Point p, Line L) // 求二维平面上点到直线的距离 
{ 
    return ( fabs((p - L.p1) * (L.p2 - L.p1)) / Norm(L.p2 - L.p1) ); 
} 
double Distance(Point3D p, Line3D L)// 求三维空间中点到直线的距离 
{ 
    return ( Norm((p - L.p1) * (L.p2 - L.p1)) / Norm(L.p2 - L.p1) ); 
} 
bool OnLine(Point p, Line L) // 判断二维平面上点p是否在直线L上 
{ 
    return ZERO( (p - L.p1) * (L.p2 - L.p1) ); 
} 
bool OnLine(Point3D p, Line3D L) // 判断三维空间中点p是否在直线L上 
{ 
    return ZERO( (p - L.p1) * (L.p2 - L.p1) ); 
} 
int Relation(Point p, Line L) // 计算点p与直线L的相对关系 ,返回ONLINE,LEFT,RIGHT
{ 
    double res = (L.p2 - L.p1) * (p - L.p1); 
    if (EQ(res, 0)) { 
        return ONLINE; 
    } else if (res > 0) { 
        return LEFT; 
    } else { 
        return RIGHT; 
    } 
} 
bool SameSide(Point p1, Point p2, Line L) // 判断点p1, p2是否在直线L的同侧 
{ 
    double m1 = (p1 - L.p1) * (L.p2 - L.p1); 
    double m2 = (p2 - L.p1) * (L.p2 - L.p1); 
    return GT(m1 * m2, 0); 
} 
bool OnLineSeg(Point p, Line L) // 判断二维平面上点p是否在线段l上 
{ 
    return ( ZERO( (L.p1 - p) * (L.p2 - p) ) && 
        LEQ((p.x - L.p1.x)*(p.x - L.p2.x), 0) && 
        LEQ((p.y - L.p1.y)*(p.y - L.p2.y), 0) ); 
} 
bool OnLineSeg(Point3D p, Line3D L) // 判断三维空间中点p是否在线段l上 
{ 
    return ( ZERO((L.p1 - p) * (L.p2 - p)) && 
        EQ( Norm(p - L.p1) + Norm(p - L.p2), Norm(L.p2 - L.p1)) ); 
} 
Point SymPoint(Point p, Line L) // 求二维平面上点p关于直线L的对称点 
{ 
    Point result; 
    double a = L.p2.x - L.p1.x; 
    double b = L.p2.y - L.p1.y; 
    double t = ( (p.x - L.p1.x) * a + (p.y - L.p1.y) * b ) / (a*a + b*b); 
    result.x = 2 * L.p1.x + 2 * a * t - p.x; 
    result.y = 2 * L.p1.y + 2 * b * t - p.y; 
    return result; 
} 
bool Coplanar(Points3D points) // 判断一个点集中的点是否全部共面 
{ 
    int i; 
    Point3D p; 
 
 
    if (points.size() < 4) return true; 
    p = (points[2] - points[0]) * (points[1] - points[0]); 
    for (i = 3; i < points.size(); i++) { 
        if (! ZERO(p & points[i]) ) return false; 
    } 
    return true; 
} 
bool LineIntersect(Line L1, Line L2) // 判断二维的两直线是否相交 
{ 
    return (! ZERO((L1.p1 - L1.p2)*(L2.p1 - L2.p2)) );  // 是否平行 
} 
bool LineIntersect(Line3D L1, Line3D L2) // 判断三维的两直线是否相交 
{ 
    Point3D p1 = L1.p1 - L1.p2; 
    Point3D p2 = L2.p1 - L2.p2; 
    Point3D p  = p1 * p2; 
    if (ZERO(p)) return false;      // 是否平行 
    p = (L2.p1 - L1.p2) * (L1.p1 - L1.p2); 
    return ZERO(p & L2.p2);         // 是否共面 
} 
bool LineSegIntersect(Line L1, Line L2) // 判断二维的两条线段是否相交 
{ 
    return ( GEQ( max(L1.p1.x, L1.p2.x), min(L2.p1.x, L2.p2.x) ) && 
        GEQ( max(L2.p1.x, L2.p2.x), min(L1.p1.x, L1.p2.x) ) && 
        GEQ( max(L1.p1.y, L1.p2.y), min(L2.p1.y, L2.p2.y) ) && 
        GEQ( max(L2.p1.y, L2.p2.y), min(L1.p1.y, L1.p2.y) ) && 
        LEQ( ((L2.p1 - L1.p1) * (L1.p2 - L1.p1)) * ((L2.p2 -  L1.p1) * (L1.p2 - L1.p1)), 0 ) && 
        LEQ( ((L1.p1 - L2.p1) * (L2.p2 - L2.p1)) * ((L1.p2 -  L2.p1) * (L2.p2 - L2.p1)), 0 ) );              
} 
bool LineSegIntersect(Line3D L1, Line3D L2) // 判断三维的两条线段是否相交 
{ 
    // todo 
    return true; 
} 
// 计算两条二维直线的交点，结果在参数P中返回 
// 返回值说明了两条直线的位置关系:  COLINE   -- 共线  PARALLEL -- 平行  CROSS    -- 相交 
int CalCrossPoint(Line L1, Line L2, Point& P) 
{ 
    double A1, B1, C1, A2, B2, C2; 
 
 
    A1 = L1.p2.y - L1.p1.y; 
    B1 = L1.p1.x - L1.p2.x; 
    C1 = L1.p2.x * L1.p1.y - L1.p1.x * L1.p2.y; 
 
 
    A2 = L2.p2.y - L2.p1.y; 
    B2 = L2.p1.x - L2.p2.x; 
    C2 = L2.p2.x * L2.p1.y - L2.p1.x * L2.p2.y; 
 
 
    if (EQ(A1 * B2, B1 * A2))    { 
        if (EQ( (A1 + B1) * C2, (A2 + B2) * C1 )) { 
            return COLINE; 
        } else { 
            return PARALLEL; 
        } 
    } else { 
        P.x = (B2 * C1 - B1 * C2) / (A2 * B1 - A1 * B2); 
        P.y = (A1 * C2 - A2 * C1) / (A2 * B1 - A1 * B2); 
        return CROSS; 
    } 
} 
// 计算两条三维直线的交点，结果在参数P中返回 
// 返回值说明了两条直线的位置关系 COLINE   -- 共线  PARALLEL -- 平行  CROSS    -- 相交  NONCOPLANAR -- 不公面 
int CalCrossPoint(Line3D L1, Line3D L2, Point3D& P) 
{ 
    // todo 
    return 0; 
} 
// 计算点P到直线L的最近点 
Point NearestPointToLine(Point P, Line L)  
{ 
    Point result; 
    double a, b, t; 
 
 
    a = L.p2.x - L.p1.x; 
    b = L.p2.y - L.p1.y; 
    t = ( (P.x - L.p1.x) * a + (P.y - L.p1.y) * b ) / (a * a + b * b); 
 
 
    result.x = L.p1.x + a * t; 
    result.y = L.p1.y + b * t; 
    return result; 
} 
// 计算点P到线段L的最近点 
Point NearestPointToLineSeg(Point P, Line L)  
{ 
    Point result; 
    double a, b, t; 
 
 
    a = L.p2.x - L.p1.x; 
    b = L.p2.y - L.p1.y; 
    t = ( (P.x - L.p1.x) * a + (P.y - L.p1.y) * b ) / (a * a + b * b); 
 
 
    if ( GEQ(t, 0) && LEQ(t, 1) ) { 
        result.x = L.p1.x + a * t; 
        result.y = L.p1.y + b * t; 
    } else { 
        if ( Norm(P - L.p1) < Norm(P - L.p2) ) { 
            result = L.p1; 
        } else { 
            result = L.p2; 
        } 
    } 
    return result; 
} 
// 计算险段L1到线段L2的最短距离 
double MinDistance(Line L1, Line L2)  
{ 
    double d1, d2, d3, d4; 
 
 
    if (LineSegIntersect(L1, L2)) { 
        return 0; 
    } else { 
        d1 = Norm( NearestPointToLineSeg(L1.p1, L2) - L1.p1 ); 
        d2 = Norm( NearestPointToLineSeg(L1.p2, L2) - L1.p2 ); 
        d3 = Norm( NearestPointToLineSeg(L2.p1, L1) - L2.p1 ); 
        d4 = Norm( NearestPointToLineSeg(L2.p2, L1) - L2.p2 ); 
         
        return min( min(d1, d2), min(d3, d4) ); 
    } 
} 
// 求二维两直线的夹角， 
// 返回值是0~Pi之间的弧度 
double Inclination(Line L1, Line L2) 
{ 
    Point u = L1.p2 - L1.p1; 
    Point v = L2.p2 - L2.p1; 
    return acos( (u & v) / (Norm(u)*Norm(v)) ); 
} 
// 求三维两直线的夹角， 
// 返回值是0~Pi之间的弧度 
double Inclination(Line3D L1, Line3D L2) 
{ 
    Point3D u = L1.p2 - L1.p1; 
    Point3D v = L2.p2 - L2.p1; 
    return acos( (u & v) / (Norm(u)*Norm(v)) ); 
} 
/
 
 
/
// 判断两个矩形是否相交 
// 如果相邻不算相交 
bool Intersect(Rect_2 r1, Rect_2 r2) 
{ 
    return ( max(r1.xl, r2.xl) < min(r1.xh, r2.xh) && 
             max(r1.yl, r2.yl) < min(r1.yh, r2.yh) ); 
} 
// 判断矩形r2是否可以放置在矩形r1内 
// r2可以任意地旋转 
//发现原来的给出的方法过不了OJ上的无归之室这题，
//所以用了自己的代码
bool IsContain(Rect r1, Rect r2)      //矩形的w>h
 { 
     if(r1.w >r2.w && r1.h > r2.h) return true;
     else
     {
        double r = sqrt(r2.w*r2.w + r2.h*r2.h) / 2.0;
        double alpha = atan2(r2.h,r2.w);
        double sita = asin((r1.h/2.0)/r);
        double x = r * cos(sita - 2*alpha);
        double y = r * sin(sita - 2*alpha);
        if(x < r1.w/2.0 && y < r1.h/2.0 && x > 0 && y > -r1.h/2.0) return true;
        else return false;
     }
} 
 
 
 
 
//圆
Point Center(const Circle & C) //圆心
{      
    return C.c;      
}    
 
 
double Area(const Circle &C)
{
 return pi*C.r*C.r; 
} 
 
 
double CommonArea(const Circle & A, const Circle & B) //两个圆的公共面积       
{      
    double area = 0.0;      
    const Circle & M = (A.r > B.r) ? A : B;      
    const Circle & N = (A.r > B.r) ? B : A;      
    double D = Distance(Center(M), Center(N));      
    if ((D < M.r + N.r) && (D > M.r - N.r))      
    {      
        double cosM = (M.r * M.r + D * D - N.r * N.r) / (2.0 * M.r * D);      
        double cosN = (N.r * N.r + D * D - M.r * M.r) / (2.0 * N.r * D);      
        double alpha = 2.0 * acos(cosM);      
        double beta  = 2.0 * acos(cosN);      
        double TM = 0.5 * M.r * M.r * sin(alpha);      
        double TN = 0.5 * N.r * N.r * sin(beta);      
        double FM = (alpha / (2*pi)) * Area(M);      
        double FN = (beta / (2*pi)) * Area(N);      
        area = FM + FN - TM - TN;      
    }      
    else if (D <= M.r - N.r)      
    {      
        area = Area(N);      
    }      
    return area;      
} 
     
bool IsInCircle(const Circle & C, const Rect_2 & rect)//判断圆是否在矩形内(不允许相切)
{      
    return (GT(C.c.x - C.r, rect.xl)
  &&  LT(C.c.x + C.r, rect.xh)
  &&  GT(C.c.y - C.r, rect.yl)
  &&  LT(C.c.y + C.r, rect.yh));      
}  
 
 
//判断2圆的位置关系
//返回: 
//BAOHAN   = 1;        // 大圆包含小圆
//NEIQIE   = 2;        // 内切
//XIANJIAO = 3;        // 相交
//WAIQIE   = 4;        // 外切
//XIANLI   = 5;        // 相离
int CirCir(const Circle &c1, const Circle &c2)//判断2圆的位置关系
{
 double dis = Distance(c1.c,c2.c);
 if(LT(dis,fabs(c1.r-c2.r))) return BAOHAN;
 if(EQ(dis,fabs(c1.r-c2.r))) return NEIQIE;
 if(LT(dis,c1.r+c2.r) && GT(dis,fabs(c1.r-c2.r))) return XIANJIAO;
 if(EQ(dis,c1.r+c2.r)) return WAIQIE;
 return XIANLI;
}
 
 
 
int main()
{
 return 0;
}
```

### 5.结构体表示几何图形all

```cpp
//计算几何(二维)   
#include <cmath>   
#include <cstdio>   
#include <algorithm>   
using namespace std;   
 
 
typedef double TYPE;   
#define Abs(x) (((x)>0)?(x):(-(x)))   
#define Sgn(x) (((x)<0)?(-1):(1))   
#define Max(a,b) (((a)>(b))?(a):(b))   
#define Min(a,b) (((a)<(b))?(a):(b))   
#define Epsilon 1e-8   
#define Infinity 1e+10   
#define PI acos(-1.0)//3.14159265358979323846   
TYPE Deg2Rad(TYPE deg){return (deg * PI / 180.0);}   
TYPE Rad2Deg(TYPE rad){return (rad * 180.0 / PI);}   
TYPE Sin(TYPE deg){return sin(Deg2Rad(deg));}   
TYPE Cos(TYPE deg){return cos(Deg2Rad(deg));}   
TYPE ArcSin(TYPE val){return Rad2Deg(asin(val));}   
TYPE ArcCos(TYPE val){return Rad2Deg(acos(val));}   
TYPE Sqrt(TYPE val){return sqrt(val);}  
 
 
//点   
struct POINT   
{   
  TYPE x;   
  TYPE y;   
  POINT() : x(0), y(0) {};   
  POINT(TYPE _x_, TYPE _y_) : x(_x_), y(_y_) {};   
};   
// 两个点的距离   
TYPE Distance(const POINT & a, const POINT & b)   
{   
  return Sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));   
}   
//线段   
struct SEG   
{     
  POINT a; //起点   
  POINT b; //终点   
  SEG() {};   
  SEG(POINT _a_, POINT _b_):a(_a_),b(_b_) {};   
};     
//直线(两点式)   
struct LINE   
{   
  POINT a;   
  POINT b;   
  LINE() {};   
  LINE(POINT _a_, POINT _b_) : a(_a_), b(_b_) {};   
};   
//直线(一般式)   
struct LINE2   
{   
  TYPE A,B,C;   
  LINE2() {};   
  LINE2(TYPE _A_, TYPE _B_, TYPE _C_) : A(_A_), B(_B_), C(_C_) {};   
};   
 
 
//两点式化一般式   
LINE2 Line2line(const LINE & L) // y=kx+c k=y/x
{   
  LINE2 L2;   
  L2.A = L.b.y - L.a.y;   
  L2.B = L.a.x - L.b.x;   
  L2.C = L.b.x * L.a.y - L.a.x * L.b.y;   
  return L2;   
}   
 
 
// 引用返回直线 Ax + By + C =0 的系数   
void Coefficient(const LINE & L, TYPE & A, TYPE & B, TYPE & C)   
{   
  A = L.b.y - L.a.y;   
  B = L.a.x - L.b.x;   
  C = L.b.x * L.a.y - L.a.x * L.b.y;   
}   
void Coefficient(const POINT & p,const TYPE a,TYPE & A,TYPE & B,TYPE & C)   
{   
  A = Cos(a);   
  B = Sin(a);   
  C = - (p.y * B + p.x * A);   
}   
/判等(值，点，直线)   
bool IsEqual(TYPE a, TYPE b)   
{   
  return (Abs(a - b) <Epsilon);   
}   
bool IsEqual(const POINT & a, const POINT & b)   
{   
  return (IsEqual(a.x, b.x) && IsEqual(a.y, b.y));   
}   
bool IsEqual(const LINE & A, const LINE & B)   
{   
  TYPE A1, B1, C1;   
  TYPE A2, B2, C2;   
  Coefficient(A, A1, B1, C1);   
  Coefficient(B, A2, B2, C2);   
  return IsEqual(A1 * B2, A2 * B1) && IsEqual(A1 * C2, A2 * C1) && IsEqual(B1 * C2, B2 * C1);   
}   
// 矩形   
struct RECT   
{   
  POINT a; // 左下点     
  POINT b; // 右上点     
  RECT() {};   
  RECT(const POINT & _a_, const POINT & _b_) { a = _a_; b = _b_; }   
};   
 
 
//矩形化标准   
RECT Stdrect(const RECT & q)
{   
  TYPE t;   
  RECT p=q;   
  if(p.a.x > p.b.x) swap(p.a.x , p.b.x);    
  if(p.a.y > p.b.y) swap(p.a.y , p.b.y);    
  return p;   
}   
 
 
//根据下标返回矩形的边     
SEG Edge(const RECT & rect, int idx)   
{   
  SEG edge;   
  while (idx < 0) idx += 4;   
  switch (idx % 4)   
  {   
  case 0: //下边
    edge.a = rect.a;   
    edge.b = POINT(rect.b.x, rect.a.y);   
    break;   
  case 1: //右边
    edge.a = POINT(rect.b.x, rect.a.y);   
    edge.b = rect.b;   
    break;   
  case 2: //上边  
    edge.a = rect.b;   
    edge.b = POINT(rect.a.x, rect.b.y);   
    break;   
  case 3: //左边  
    edge.a = POINT(rect.a.x, rect.b.y);   
    edge.b = rect.a;   
    break;   
  default:   
    break;   
  }   
  return edge;   
}   
 
 
//矩形的面积   
TYPE Area(const RECT & rect)   
{   
  return (rect.b.x - rect.a.x) * (rect.b.y - rect.a.y);   
}   
 
 
//两个矩形的公共面积     
TYPE CommonArea(const RECT & A, const RECT & B)   
{   
  TYPE area = 0.0;   
  POINT LL(Max(A.a.x, B.a.x), Max(A.a.y, B.a.y));   
  POINT UR(Min(A.b.x, B.b.x), Min(A.b.y, B.b.y));   
  if( (LL.x <= UR.x) && (LL.y <= UR.y) )   
  {   
    area = Area(RECT(LL, UR));   
  }   
  return area;   
}  
//判断圆是否在矩形内(不允许相切)   
bool IsInCircle(const CIRCLE & circle, const RECT & rect)   
{   
  return (circle.x - circle.r > rect.a.x) &&   
    (circle.x + circle.r < rect.b.x) &&   
    (circle.y - circle.r > rect.a.y) &&   
    (circle.y + circle.r < rect.b.y);   
}   
 
 
//判断矩形是否在圆内(不允许相切)   
bool IsInRect(const CIRCLE & circle, const RECT & rect)   
{   
  POINT c,d;   
  c.x=rect.a.x; c.y=rect.b.y;   
  d.x=rect.b.x; d.y=rect.a.y;   
  return (Distance( Center(circle) , rect.a ) < circle.r) &&   
    (Distance( Center(circle) , rect.b ) < circle.r) &&   
    (Distance( Center(circle) , c ) < circle.r) &&   
    (Distance( Center(circle) , d ) < circle.r);   
}   
 
 
//判断矩形是否与圆相离(不允许相切)   
bool Isoutside(const CIRCLE & circle, const RECT & rect)   
{   
  POINT c,d;   
  c.x=rect.a.x; c.y=rect.b.y;   
  d.x=rect.b.x; d.y=rect.a.y;   
  return (Distance( Center(circle) , rect.a ) > circle.r) &&   
    (Distance( Center(circle) , rect.b ) > circle.r) &&   
    (Distance( Center(circle) , c ) > circle.r) &&   
    (Distance( Center(circle) , d ) > circle.r) &&   
    (rect.a.x > circle.x || circle.x > rect.b.x || rect.a.y > circle.y || circle.y > rect.b.y) ||   
    ((circle.x - circle.r > rect.b.x) ||   
    (circle.x + circle.r < rect.a.x) ||   
    (circle.y - circle.r > rect.b.y) ||   
    (circle.y + circle.r < rect.a.y));   
}   
```

## 6.四城部分几何模板

```cpp
/*
1.注意实际运用的时候可以用sqrd代替dist提高精度，节省时间
*/
#include <iostream>
#include <math.h>
#include <algorithm>
using namespace std;
 
 
const double INF = 10e300;
const double EPS = 1e-8;
const double PI = acos(-1.0);
 
 
inline int dblcmp(double a, double b) {if(fabs(a-b) < EPS) return 0;if(a < b) return -1;return 1;}
inline double Max(double a, double b) { if(dblcmp(a, b) == 1) return a; return b; }
inline double Min(double a, double b) { if(dblcmp(a, b) == 1) return b; return a; }
inline double Agl(double deg) { return deg * PI / 180.0; }
 
 
struct Point { double x, y; void set(double a, double b) { x = a; y = b; } };
struct Vec { double x, y; void set(Point& a, Point& b) { x = b.x-a.x; y = b.y-a.y; } };
struct Line { double a, b, c; Point st, end;
void set(Point& u, Point& v) {a = v.y - u.y; b = u.x - v.x; c = a*u.x + b*u.y; st = u; end = v; } };
 
 
inline double dist(Point& a, Point& b) { return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)); }
inline double sqrd(Point& a, Point& b) { return (a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y); }
inline double dot(Vec& a, Vec& b) { return a.x * b.x + a.y * b.y; }
inline double cross(Vec& a, Vec& b) { return a.x * b.y - a.y * b.x; }
inline double cross(Point& a, Point& b, Point& c) {Vec x, y; x.set(a, b); y.set(a, c); return cross(x, y); }
//返回1代表a在bc之间 0代表在端点 -1代表在外面
inline int between(Point& a, Point& b, Point& c) { Vec x, y; x.set(a,b); y.set(a,c); return dblcmp(dot(x, y),0); }
 
 
//3维坐标转换 输入是度数
void trans(double lat, double log, double& x, double& y, double& z, double radius) {
	x = radius * cos(lat) * cos(log);
	y = radius * cos(lat) * sin(log);
	z = radius * sin(lat);
}
 
 
//求两点的平分线
Line bisector(Point& a, Point& b) {
	Line ab, ans; ab.set(a, b);
	double midx = (a.x + b.x)/2.0,	midy = (a.y + b.y)/2.0;
	ans.a = -ab.b, ans.b = -ab.a, ans.c = -ab.b * midx + ab.a * midy;
	return ans;
}
 
 
 
 
//线线相交 如果平行 返回-1, 重合返回-2
int line_line_intersect(Line& l1, Line& l2, Point& s) {
	double det = l1.a*l2.b - l2.a*l1.b;
    if(dblcmp(det, 0.0) == 0) { //平行或者重合
		if(dblcmp(point_line_dist(l1.st, l2.st, l2.end, 0), 0) == 0) 
			return -2;
		return -1;
	}
    s.x = (l2.b*l1.c - l1.b*l2.c)/det;
    s.y = (l1.a*l2.c - l2.a*l1.c)/det;
	return 1;
}
 
 
//2线段相交 ab, cd 交点是s 平行返回-1, 重合返回-2, 不在线段上面返回0 在线段中间返回1 在线段两端返回2
int seg_seg_intersect(Point& a, Point& b, Point& c, Point& d, Point& s) {
    Line l1, l2; l1.set(a, b); l2.set(c, d);
	int ans = line_line_intersect(l1, l2, s);
	if(ans == 1) {
		if(between(s, a, b) == 1 && between(s, c, d) == 1) 
			return 1;
		if(between(s, a, b) == -1 && between(s, c, d) == -1)
			return 0;
		return 2;
	}
	return ans;
}
 
 
//求三点共圆 中心放在center中 返回半径
double center_3point(Point& a, Point& b, Point& c, Point& center) {
	Line x = bisector(a, b), y = bisector(b, c);
	line_line_intersect(x, y, center);
	return dist(center, a);
}
```

## 7.经典题目

### 最近点对（UOJ）

```
#include <bits/stdc++.h>
 const int N=2e6+10;
struct POINT 
{
    long double x,y;
}point[N],temp[N];
 
long double dis(struct POINT p1, struct POINT p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}
 
int cmp(const void * a, const void * b)
{
    struct POINT * c = (struct POINT *)a;
    struct POINT * d = (struct POINT *)b;
    if (c->x != d->x)
    {
        return c->x > d->x;
    }
    else
        return c->y > d->y;
}
int cmp1(const void * a, const void * b)
{
    struct POINT * c = (struct  POINT *)a;
    struct POINT * d = (struct  POINT *)b;
    if (c->y != d->y)
    {
        return c->y > d->y;
    }
    else
        return c->x > d->x;
}
 
long double findMin(int l, int r)
{
    if (l == r)
    {
        return 10010;
    }
    if (l == r - 1)
    {
        return dis(point[l], point[r]);
    }
    long double tmp1 = findMin(l,(l + r) >> 1);
    long double tmp2 = findMin(((l + r) >> 1) + 1, r);
    long double Mindis,tmp, mid;
    mid = point[(l + r) >> 1].x;
    /*mid = (point[l].x + point[r].x) / 2.0;*/
    int i,j,cnt = 0;
    if (tmp1 < tmp2)
    {
        Mindis = tmp1;
    }
    else
        Mindis = tmp2;
    for (i = l; i <= r; ++ i)
    {
        if (fabs(point[i].x - mid) < Mindis)
        {
            temp[cnt ++] = point[i];
        }
    }
    qsort(temp, cnt, sizeof(temp[0]), cmp1);
    for (i = 0; i < cnt - 1; ++ i)
    {
        /*for (j = i + 1; j < cnt; ++ j)*/
        for (j = i + 1; j < i + 7 && j < cnt; ++ j)
        {
            tmp = dis(temp[i], temp[j]);
            if (tmp < Mindis)
            {
                Mindis = tmp;
            }
        }
    }
    return Mindis;
 
}
int main()
{
    int n,i,j;
    long double minDis;
    while (scanf("%d", &n)==1 && n)
    {
        for (i = 0; i < n; ++ i)
        {
            scanf("%lf%lf", &point[i].x, &point[i].y);
        }
        qsort(point, n, sizeof(point[0]), cmp);
        minDis = findMin(0, n-1);
        if (minDis > 10000)
        {
            printf("INFINITY\n");
        }
        else
            printf("%.4lf\n", minDis);
    }
    return 0;
}
```

### 洛谷加强加强版本 最近点对

```
#include<bits/stdc++.h>
using namespace std;
#define ll long long
const int INF =2147483647;
struct Point{int x,y;};
typedef vector<Point>::iterator Iter;
bool cmpx(const Point a,const Point b){return a.x<b.x;}
bool cmpy(const Point a,const Point b){return a.y<b.y;}
double dis(const Point a,const Point b){
    return sqrt(pow(a.x-b.x,2)+pow(a.y-b.y,2));
}
void slv(const Iter l,const Iter r,double &d){
    if(r-l<=1) return;
    vector<Point> Q; Iter t=l+(r-l)/2;double w=t->x;
    slv(l,t,d),slv(t,r,d),inplace_merge(l,t,r,cmpy);
    for(Iter x=l;x!=r;++x)
        if(abs(w-x->x)<=d) Q.push_back(*x);
    for(Iter x=Q.begin(),y=x;x!=Q.end();++x){
        while(y!=Q.end()&&y->y<=x->y+d) ++y;
        for(Iter z=x+1;z!=y;++z) d=min(d,dis(*x,*z));
    }
}
vector<Point> X; int n; double ans=1e18;

int main(){
	ios::sync_with_stdio(false);
    cin.tie(0);cout.tie(0);
    cin>>n; 
    for(int i=0;i<n;i++)
    {
    	int x,y;
        cin>>x>>y; 
        X.push_back({x,y});
    }
    sort(X.begin(),X.end(),cmpx),slv(X.begin(),X.end(),ans);
    printf("%.4lf\n",ans);
    return 0;
}
```



### 2023CCPC网络赛 gap（法向量 投影）

```
#include <iostream>
#include<cmath>
#include<vector>
#include<iomanip>
using namespace std;
const int N = 53;
const double INF = 1e18;
const double eps = 1e-8;
#define zero(x) (((x) > 0 ? (x) : -(x)) < eps)
struct point3
{
    double x, y, z;
    point3 operator+(const point3 &o) const
    {
        return {x + o.x, y + o.y, z + o.z};
    }
    point3 operator-(const point3 &o) const
    {
        return {x - o.x, y - o.y, z - o.z};
    }
    point3 operator*(const double &o) const
    {
        return {x*o , y *o, z *o};
    }
    point3 operator/(const double &o) const
    {
        return {x/o , y /o, z /o};
    }
    bool operator<(const point3 &o) const
    {
        if (!zero(x - o.x))
            return x < o.x;
        if (!zero(y - o.y))
            return y < o.y;
        return z < o.z;
    }
    bool operator!=(const point3 &o) const
    {
        return (!zero(x - o.x) || !zero(y - o.y) || !zero(z - o.z));
    }
}a[N];
vector<point3> line;
double vlen(point3 p)
{
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}
point3 xmult(point3 u, point3 v)
{
    point3 ret;
    ret.x = u.y * v.z - v.y * u.z;
    ret.y = u.z * v.x - u.x * v.z;
    ret.z = u.x * v.y - u.y * v.x;
    return ret;
}
double dmult(point3 u, point3 v)
{
    return u.x * v.x + u.y * v.y + u.z * v.z;
}
point3 projection(point3 p, point3 u)
{
    return u*dmult(p,u);
}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    int n;
    cin >> n;
    for (int i = 0; i < n; ++i)
        cin >> a[i].x >> a[i].y >> a[i].z;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (a[i] != a[j])
                line.push_back(a[i]-a[j]);
    double ans = INF;
    for (int i = 0; i < line.size(); ++i)
    {
        point3 A = line[i];
        for (int j = i + 1; j < line.size(); ++j)
        {
            point3 B = line[j];
            if (zero(fabs(dmult(A, B)/vlen(A)/vlen(B))-1))
                continue;
            point3 normalVector = xmult(A, B);
            normalVector=normalVector/vlen(normalVector);
            point3 mi = {INF, INF, INF}, ma = {-INF, -INF, -INF};
            for (int k = 0; k < n; ++k)
            {
                point3 res = projection(a[k], normalVector);
                if (res < mi)
                    mi = res;
                if (ma < res)
                    ma = res;
            }
            ans = min(ans, vlen(mi-ma));
        }
    }
    if (ans == INF || zero(ans))
        ans = 0;
    cout << fixed << setprecision(15) << ans;
    return 0;
}
```



### 直线旋转_两凸包的最短距离

```cpp
#include <stdio.h>
#include <math.h>
#define pi acos(-1.0)
#define eps 1e-6
#define inf 1e250
#define Maxn 10005
typedef struct TPoint
{
	double x, y;
}TPoint;
typedef struct TPolygon
{
	TPoint p[Maxn];
	int n;
}TPolygon;
typedef struct TLine
{
	double a, b, c;
}TLine;
 
double max(double a, double b)
{
	if(a > b) return a;
	return b;
}
 
double min(double a, double b)
{
	if(a < b) return a;
	return b;
}
 
double distance(TPoint p1, TPoint p2)
{
	return sqrt((p1.x - p2.x) * (p1.x - p2.x) 
	 + (p1.y - p2.y) * (p1.y - p2.y));
}
 
TLine lineFromSegment(TPoint p1, TPoint p2)
{
    TLine tmp;
    tmp.a = p2.y - p1.y;
    tmp.b = p1.x - p2.x;
    tmp.c = p2.x * p1.y - p1.x * p2.y;
    return tmp;
}
 
double polygonArea(TPolygon p)
{
    int i, n;
    double area;
    n = p.n;
    area = 0;
    for(i = 1;i <= n;i++)
		area += (p.p[i - 1].x * p.p[i % n].y - p.p[i % n].x * p.p[i - 1].y);
 
	return area / 2;  
}
 
void ChangeClockwise(TPolygon &polygon)
{
	TPoint tmp;
	int i;
	for(i = 0;i <= (polygon.n - 1) / 2;i++)
	{
		tmp = polygon.p[i];
		polygon.p[i] = polygon.p[polygon.n - 1 - i];
		polygon.p[polygon.n - 1 - i] = tmp;			
	}
}
 
double disPointToSeg(TPoint p1, TPoint p2, TPoint p3)
{
    double a = distance(p1, p2);
    double b = distance(p1, p3);
    double c = distance(p2, p3);
    if(fabs(a + b - c) < eps) return 0;
    if(fabs(a + c - b) < eps || fabs(b + c - a) < eps) return min(a, b);
    double t1 = -a * a + b * b + c * c;
    double t2 = a * a - b * b + c * c;
    if(t1 <= 0 || t2 <= 0) return min(a, b);
    
    TLine l1 = lineFromSegment(p2, p3);
    return fabs(l1.a * p1.x + l1.b * p1.y + l1.c) / sqrt(l1.a * l1.a + l1.b * l1.b);   
}
 
double disPallSeg(TPoint p1, TPoint p2, TPoint p3, TPoint p4)
{
	return min(min(disPointToSeg(p1, p3, p4), disPointToSeg(p2, p3, p4)),
	 min(disPointToSeg(p3, p1, p2), disPointToSeg(p4, p1, p2)));
}
 
 
double angle(TPoint p1, TPoint p2, double SlewRate)
{
	double ang, tmp;
	TPoint p;
	p.x = p2.x - p1.x;
	p.y = p2.y - p1.y;
	if(fabs(p.x) < eps)
	{
		if(p.y > 0) ang = pi / 2;
		else ang = 3 * pi / 2;
	}
	else 
	{
        ang = atan(p.y / p.x);
		if(p.x < 0) ang += pi;
	}
	while(ang < 0) ang += 2 * pi;
	if(ang >= pi) SlewRate += pi;
	if(ang > SlewRate) tmp = ang - SlewRate;
	else tmp = pi - (SlewRate - ang);
	while(tmp >= pi) tmp -= pi;
	if(fabs(tmp - pi) < eps) tmp = 0;
	return tmp;
}
 
 
int main()
{
	int n, m, i;
	TPolygon polygon1, polygon2;
	double ymin1, ymax2, ans, d;
	int k1, k2;
	while(scanf("%d%d", &n, &m) && n)
	{
		polygon1.n = n;
		polygon2.n = m;
		for(i = 0;i < n;i++)
			scanf("%lf%lf", &polygon1.p[i].x, &polygon1.p[i].y);
		for(i = 0;i < m;i++)
			scanf("%lf%lf", &polygon2.p[i].x, &polygon2.p[i].y);	
		if(polygonArea(polygon1) < 0) ChangeClockwise(polygon1);
		if(polygonArea(polygon2) < 0) ChangeClockwise(polygon2);
		ymin1 = inf, ymax2 = -inf;
		for(i = 0;i < n;i++)
			if(polygon1.p[i].y < ymin1) ymin1 = polygon1.p[i].y , k1 = i;
		for(i = 0;i < m;i++)
			if(polygon2.p[i].y > ymax2) ymax2 = polygon2.p[i].y , k2 = i;	
		double SlewRate = 0;
		double angle1, angle2;
		ans = inf;
		double Slope = 0;
		while(Slope <= 360)	
		{	
			while(SlewRate >= pi) SlewRate -= pi;
			if(fabs(pi - SlewRate) < eps) SlewRate = 0;
			angle1 = angle(polygon1.p[k1], polygon1.p[(k1 + 1) % n], SlewRate);
			angle2 = angle(polygon2.p[k2], polygon2.p[(k2 + 1) % m], SlewRate);	
			if(fabs(angle1 - angle2) < eps)
			{
				d = disPallSeg(polygon1.p[k1], polygon1.p[(k1 + 1) % n], polygon2.p[k2], polygon2.p[(k2 + 1) % m]); 
				if(d < ans) ans = d;
                k1++;
				k1 %= n;
				k2++;
				k2 %= m; 
				SlewRate += angle1;
				Slope += angle1;
			}
			else if(angle1 < angle2)
			{
				d = disPointToSeg(polygon2.p[k2], polygon1.p[k1], polygon1.p[(k1 + 1) % n]);
				if(d < ans) ans = d;
				k1++;
				k1 %= n;
				SlewRate += angle1;
				Slope += angle1;
			}
			else 
			{
				d = disPointToSeg(polygon1.p[k1], polygon2.p[k2], polygon2.p[(k2 + 1) % m]);
				if(d < ans) ans = d;
				k2++;
				k2 %= m;
				SlewRate += angle2;
				Slope += angle2;
			}
		}
		printf("%.5lf\n", ans);
	}
	return 0;
}
```

### 扇形的重心

```cpp
//Xc = 2*R*sinA/3/A 
//A为圆心角的一半
#include <stdio.h>
#include <math.h>
int main()
{
	double r, angle;
	while(scanf("%lf%lf", &r, &angle) != EOF){
		angle /= 2;
		printf("%.6lf\n", 2 * r * sin(angle) / 3 / angle);
	}
	return 0;
}
```

### 存不存在一个平面把两堆点分开

```cpp
#include <stdio.h>
struct point
{
	double x, y, z;
}pa[201], pb[201];
int main() 
{ 
	int n, m, i; 
	while (scanf("%d", &n), n != -1) 
	{ 
		for (i = 0; i < n; i++) 
			scanf("%lf%lf%lf", &pa[i].x, &pa[i].y, &pa[i].z); 
		scanf("%d", &m); 
		for (i = 0; i < m; i++) 
			scanf("%lf%lf%lf", &pb[i].x, &pb[i].y, &pb[i].z);
		int cnt = 0, finish = 0; 
		double a = 0, b = 0, c = 0, d = 0; 
		while (cnt < 100000 && !finish)
		{ 
			finish = 1; 
			for (i = 0; i < n; i++) 
				if (a * pa[i].x + b * pa[i].y + c * pa[i].z + d > 0) 
				{ 
					a -= pa[i].x; 
					b -= pa[i].y; 
					c -= pa[i].z; 
					d -= 3; 
					finish = 0; 
				}
			for (i = 0; i < m; i++) 
				if (a * pb[i].x + b * pb[i].y + c * pb[i].z + d <= 0) 
				{ 
					a += pb[i].x; 
					b += pb[i].y; 
					c += pb[i].z; 
					d += 3; 
					finish = 0; 
				}
			cnt++; 
		}
		printf("%lf %lf %lf %lf\n", a, b, c, d); 
	}
	return 0;
}
```

### 共线最多的点的个数

```cpp
/*
2617120 chenhaifeng 1118 Accepted 512K 1890MS C++ 977B 2007-09-04 18:43:26 
直接O(n^3)超时，用一个标记数组，标记i,j所做直线已经查找过，可以跳过
大牛的思想
朴素做法是 O(n3) 的，超时。我的做法是枚举每个点，
然后求其它点和它连线的斜率，再排序。这样就得到经过
该点的直线最多能经过几个点。求个最大值就行了。复
杂度是 O(n2logn) 的。把排序换成 hash，
可以优化到 O(n2)。 
2617134 chenhaifeng 1118 Accepted 276K 312MS G++ 1394B 2007-09-04 18:49:08 
*/
#include <stdio.h>
#include <math.h>
 
 
bool f[705][705];
int a[705];
 
 
int main()
{
	int n, i, j, s, num, maxn;
	int x[705], y[705];
	int t, m;
 
 
	
	while(scanf("%d", &n) != EOF && n){
		for(i = 0;i <= n - 1;i++){
			scanf("%d%d", &x[i], &y[i]);
		}
		maxn = -1;
		for(i = 0;i <= n - 1;i++){
			for(j = i;j <= n - 1;j++){
				f[i][j] = false;
			}
		}
		for(i = 0;i <= n - 1;i++){
			for(j = i + 1;j <= n - 1;j++){
				if(f[i][j] == true) continue;
				if(n - j < maxn) break;
				num = 2;
				t = 2;
				a[0] = i;
				a[1] = j;
				f[i][j] = true; 
				for(s = j + 1;s <= n - 1;s++){
					if(f[i][s] == true || f[j][s] == true) continue;
					if((y[i] - y[s]) * (x[j] - x[s]) == (x[i] - x[s]) * (y[j] - y[s])){
						 num++;	
						 a[t] = s;
						 for(m = 0;m <= t - 1;m++){
								f[m][s] = true;
						}
						t++;	
					}	 			
				}
				if(num > maxn) maxn = num;	
			}			
		}
		printf("%d\n", maxn);			
	}
	return 0;
} 
```

### 线段围成的区域可储水量

```cpp
/*
两条线不相交，
左边或右边的口被遮住，
交点是某条线的那个纵坐标较高的那点
某条线段水平放置
*/
#include <stdio.h>
#include <math.h>
 
 
#define eps 1e-8
 
 
struct TPoint
{
	double x, y;
};
struct TLine
{
    double a, b, c;
};
 
 
int same(TPoint p1, TPoint p2)
{
	if(fabs(p1.x - p2.x) > eps) return 0;
	if(fabs(p1.y - p2.y) > eps) return 0;
	return 1;
}
 
 
double min(double x, double y)
{
    if(x < y) return x;
    else return y; 
}
 
 
double max(double x, double y)
{
    if(x > y) return x;
    else return y; 
}
 
 
double multi(TPoint p1, TPoint p2, TPoint p0)
{ 
    return (p1.x - p0.x) * (p2.y - p0.y) 
	     - (p2.x - p0.x) * (p1.y - p0.y);
}
 
 
bool isIntersected(TPoint s1, TPoint e1, TPoint s2, TPoint e2)
{
    if(
    (max(s1.x, e1.x) >= min(s2.x, e2.x)) &&
    (max(s2.x, e2.x) >= min(s1.x, e1.x)) &&
    (max(s1.y, e1.y) >= min(s2.y, e2.y)) &&
    (max(s2.y, e2.y) >= min(s1.y, e1.y)) &&
    (multi(s2, e1, s1) * multi(e1, e2, s1) >= 0) &&
    (multi(s1, e2, s2) * multi(e2, e1, s2) >= 0)
    )  return true;
    
    return false;    
}
 
 
TLine lineFromSegment(TPoint p1, TPoint p2)
{
    TLine tmp;
    tmp.a = p2.y - p1.y;
    tmp.b = p1.x - p2.x;
    tmp.c = p2.x * p1.y - p1.x * p2.y;
    return tmp;
}
 
 
TPoint LineInter(TLine l1, TLine l2)
{
    TPoint tmp; 
    double a1 = l1.a;
    double b1 = l1.b;
    double c1 = l1.c;
    double a2 = l2.a;
    double b2 = l2.b;
    double c2 = l2.c;
    if(fabs(b1) < eps){
        tmp.x = -c1 / a1;  
        tmp.y = (-c2 - a2 * tmp.x) / b2;
    }       
    else{
        tmp.x = (c1 * b2 - b1 * c2) / (b1 * a2 - b2 * a1);
        tmp.y = (-c1 - a1 * tmp.x) / b1;
    }
	return tmp;
}
 
 
double triangleArea(TPoint p1, TPoint p2, TPoint p3)
{
	TPoint p4, p5;
	p4.x = p2.x - p1.x;
	p4.y = p2.y - p1.y;
	p5.x = p3.x - p1.x;
	p5.y = p3.y - p1.y;
	return fabs(p5.x * p4.y - p5.y * p4.x) / 2;	
}
 
 
double find_x(double y, TLine line)
{
	return (-line.c - line.b * y) / line.a;
}
 
 
double find_y(double x, TLine line)
{
	if(fabs(line.b) < eps)
	{
		return -1e250;
	}
	else 
	{
		return (-line.c - line.a  * x) / line.b;
	}
}
 
 
int main()
{
	//freopen("in.in", "r", stdin);
	//freopen("out.out", "w", stdout);
	int test;
	double miny, y;
	TLine l1, l2;
	TPoint p1, p2, p3, p4, inter;
	TPoint tp1, tp2;
	scanf("%d", &test);
	while(test--)
	{
		scanf("%lf%lf%lf%lf%lf%lf%lf%lf", &p1.x, &p1.y, 
		&p2.x, &p2.y, &p3.x, &p3.y, &p4.x, &p4.y);
		if(same(p1, p2) || same(p3, p4) 
		   || !isIntersected(p1, p2, p3, p4)
		   || fabs(p1.y - p2.y) < eps //平行与x轴 
		   || fabs(p3.y - p4.y) < eps
		  )
		{
			printf("0.00\n");
			continue;
		}
		l1 = lineFromSegment(p1, p2);
		l2 = lineFromSegment(p3, p4);
		inter = LineInter(l1, l2);
		if(p1.y > p2.y) tp1 = p1;
		else tp1 = p2;
		if(p3.y > p4.y) tp2 = p3;
		else tp2 = p4;
		if(tp1.y < tp2.y)
		{
			if(tp1.x >= min(p4.x, p3.x) && tp1.x <= max(p4.x, p3.x))
			{
				y = find_y(tp1.x, l2);
				if(y >= tp1.y)
				{
					printf("0.00\n");
					continue;
				}	
			}
			miny = tp1.y;
		}
		else
		{
			if(tp2.x >= min(p1.x, p2.x) && tp2.x <= max(p1.x, p2.x))
			{
				y = find_y(tp2.x, l1);
				if(y >= tp2.y)
				{
					printf("0.00\n");
					continue;
				}	
			}
			miny = tp2.y;
		}
		if(fabs(miny - inter.y) < eps)
		{
			printf("0.00\n");
			continue;			
		}
		tp1.x = find_x(miny, l1);
		tp2.x = find_x(miny, l2);
		tp1.y = tp2.y = miny;
		printf("%.2lf\n", triangleArea(tp1, tp2, inter));	
	} 
	return 0;
}/*
两条线不相交，
左边或右边的口被遮住，
交点是某条线的那个纵坐标较高的那点
某条线段水平放置
*/
#include <stdio.h>
#include <math.h>
 
 
#define eps 1e-8
 
 
struct TPoint
{
	double x, y;
};
struct TLine
{
    double a, b, c;
};
 
 
int same(TPoint p1, TPoint p2)
{
	if(fabs(p1.x - p2.x) > eps) return 0;
	if(fabs(p1.y - p2.y) > eps) return 0;
	return 1;
}
 
 
double min(double x, double y)
{
    if(x < y) return x;
    else return y; 
}
 
 
double max(double x, double y)
{
    if(x > y) return x;
    else return y; 
}
 
 
double multi(TPoint p1, TPoint p2, TPoint p0)
{ 
    return (p1.x - p0.x) * (p2.y - p0.y) 
	     - (p2.x - p0.x) * (p1.y - p0.y);
}
 
 
bool isIntersected(TPoint s1, TPoint e1, TPoint s2, TPoint e2)
{
    if(
    (max(s1.x, e1.x) >= min(s2.x, e2.x)) &&
    (max(s2.x, e2.x) >= min(s1.x, e1.x)) &&
    (max(s1.y, e1.y) >= min(s2.y, e2.y)) &&
    (max(s2.y, e2.y) >= min(s1.y, e1.y)) &&
    (multi(s2, e1, s1) * multi(e1, e2, s1) >= 0) &&
    (multi(s1, e2, s2) * multi(e2, e1, s2) >= 0)
    )  return true;
    
    return false;    
}
 
 
TLine lineFromSegment(TPoint p1, TPoint p2)
{
    TLine tmp;
    tmp.a = p2.y - p1.y;
    tmp.b = p1.x - p2.x;
    tmp.c = p2.x * p1.y - p1.x * p2.y;
    return tmp;
}
 
 
TPoint LineInter(TLine l1, TLine l2)
{
    TPoint tmp; 
    double a1 = l1.a;
    double b1 = l1.b;
    double c1 = l1.c;
    double a2 = l2.a;
    double b2 = l2.b;
    double c2 = l2.c;
    if(fabs(b1) < eps){
        tmp.x = -c1 / a1;  
        tmp.y = (-c2 - a2 * tmp.x) / b2;
    }       
    else{
        tmp.x = (c1 * b2 - b1 * c2) / (b1 * a2 - b2 * a1);
        tmp.y = (-c1 - a1 * tmp.x) / b1;
    }
	return tmp;
}
 
 
double triangleArea(TPoint p1, TPoint p2, TPoint p3)
{
	TPoint p4, p5;
	p4.x = p2.x - p1.x;
	p4.y = p2.y - p1.y;
	p5.x = p3.x - p1.x;
	p5.y = p3.y - p1.y;
	return fabs(p5.x * p4.y - p5.y * p4.x) / 2;	
}
 
 
double find_x(double y, TLine line)
{
	return (-line.c - line.b * y) / line.a;
}
 
 
double find_y(double x, TLine line)
{
	if(fabs(line.b) < eps)
	{
		return -1e250;
	}
	else 
	{
		return (-line.c - line.a  * x) / line.b;
	}
}
 
 
int main()
{
	//freopen("in.in", "r", stdin);
	//freopen("out.out", "w", stdout);
	int test;
	double miny, y;
	TLine l1, l2;
	TPoint p1, p2, p3, p4, inter;
	TPoint tp1, tp2;
	scanf("%d", &test);
	while(test--)
	{
		scanf("%lf%lf%lf%lf%lf%lf%lf%lf", &p1.x, &p1.y, 
		&p2.x, &p2.y, &p3.x, &p3.y, &p4.x, &p4.y);
		if(same(p1, p2) || same(p3, p4) 
		   || !isIntersected(p1, p2, p3, p4)
		   || fabs(p1.y - p2.y) < eps //平行与x轴 
		   || fabs(p3.y - p4.y) < eps
		  )
		{
			printf("0.00\n");
			continue;
		}
		l1 = lineFromSegment(p1, p2);
		l2 = lineFromSegment(p3, p4);
		inter = LineInter(l1, l2);
		if(p1.y > p2.y) tp1 = p1;
		else tp1 = p2;
		if(p3.y > p4.y) tp2 = p3;
		else tp2 = p4;
		if(tp1.y < tp2.y)
		{
			if(tp1.x >= min(p4.x, p3.x) && tp1.x <= max(p4.x, p3.x))
			{
				y = find_y(tp1.x, l2);
				if(y >= tp1.y)
				{
					printf("0.00\n");
					continue;
				}	
			}
			miny = tp1.y;
		}
		else
		{
			if(tp2.x >= min(p1.x, p2.x) && tp2.x <= max(p1.x, p2.x))
			{
				y = find_y(tp2.x, l1);
				if(y >= tp2.y)
				{
					printf("0.00\n");
					continue;
				}	
			}
			miny = tp2.y;
		}
		if(fabs(miny - inter.y) < eps)
		{
			printf("0.00\n");
			continue;			
		}
		tp1.x = find_x(miny, l1);
		tp2.x = find_x(miny, l2);
		tp1.y = tp2.y = miny;
		printf("%.2lf\n", triangleArea(tp1, tp2, inter));	
	} 
	return 0;
}
```

### N个点最多组成多少个正方形

```cpp
#include <iostream>
#include <algorithm>

using namespace std;
const int maxn = 600;

struct Point
{
    int x, y;
    Point(){}
    Point(int x, int y):x(x), y(y){}
    bool operator<(const Point &rhs)const{
        if(this->x<rhs.x){
            return true;
        }   
        else if(this->x==rhs.x){
            return this->y > rhs.y;
        }
        else{
            return false;
        }
    }
}A[maxn];

int main(){
    int n;
    while(cin>>n){
        for(int i=0; i<n; ++i){
            cin>>A[i].x>>A[i].y;
        }
        sort(A, A+n);
        int cnt = 0;

        for(int i=0; i<n-1; ++i){
            for(int j=i+1; j<n; ++j){

                int x1 = A[i].x+(A[j].y-A[i].y); int y1 = A[i].y-(A[j].x-A[i].x);
                int x2 = A[j].x+(A[j].y-A[i].y); int y2 = A[j].y-(A[j].x-A[i].x);
                Point a(x1, y1), b(x2, y2);
                int pos1 = lower_bound(A, A+n, a) - A;
                int pos2 = lower_bound(A, A+n, b) - A;
                    if(A[pos1].x==a.x&&A[pos1].y==a.y && A[pos2].x==b.x && A[pos2].y==b.y){
                        ++cnt;
                    }

            }
        }
        cout<<cnt/2<<endl;
    }   
    return 0;
}
```

### N个点最多确定多少互不平行的直线

```cpp
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
 
 
#define eps 1e-6
#define pi acos(-1)
 
 
struct point 
{
	double x, y;
};
 
 
double FindSlewRate(point p1, point p2)
{
	point p;
	p.x = p2.x - p1.x;
	p.y = p2.y - p1.y;
	if(fabs(p.x) < eps) return pi / 2;
	double tmp = atan(p.y / p.x);
	if(tmp < 0) return pi + tmp;
	return tmp;
}
 
 
int cmp(const void *a, const void *b)
{
	double *c = (double *)a;
	double *d = (double *)b;
	if(*c < *d) return -1;
	return 1;
}
 
 
int main()
{
	int n, rt;
	point p[205];
	double rate[40005];
	while(scanf("%d", &n) != EOF)
	{
		for(int i = 0;i < n;i++)
			scanf("%lf%lf", &p[i].x ,&p[i].y);
		rt = 0;
		for(int i = 0;i < n;i++)
			for(int j = i + 1;j < n;j++)
				rate[rt++] = FindSlewRate(p[i], p[j]);
		qsort(rate, rt, sizeof(rate[0]), cmp);	
		int ans = 1;
		for(int i = 1;i < rt;i++)
			if(rate[i] > rate[i - 1]) ans++;
		//注意这里写fabs(rate[i] - rate[i - 1]) > eps Wrong Answer 
		printf("%d\n", ans);
	} 
	return 0;
}
```

#### 最大空凸包、最大空矩形(O(n^3))

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<string>
#include<algorithm>
#include<cmath>
using namespace std;

int getint()
{
    int i=0,f=1;char c;
    for(c=getchar();(c!='-')&&(c<'0'||c>'9');c=getchar());
    if(c=='-')f=-1,c=getchar();
    for(;c>='0'&&c<='9';c=getchar())i=(i<<3)+(i<<1)+c-'0';
    return i*f;
}

const int N=105;
struct point
{
    double x,y;
    point(){}
    point(double _x,double _y):x(_x),y(_y){}
    inline friend point operator - (const point &a,const point &b)
    {return point(a.x-b.x,a.y-b.y);}
    inline friend double operator * (const point &a,const point &b)
    {return a.x*b.y-a.y*b.x;}
    inline double dis(){return x*x+y*y;}
}a[N],p[N],O;
int T,n,m;
double dp[N][N],ans;

inline bool cmp(const point &a,const point &b)
{
    double res=(a-O)*(b-O);
    if(res)return res>0;
    return (a-O).dis()<(b-O).dis();
}

void solve()
{
    memset(dp,0,sizeof(dp));
    sort(p+1,p+m+1,cmp);
    for(int i=1;i<=m;i++)
    {
        int j=i-1;
        while(j&&!((p[i]-O)*(p[j]-O)))j--;
        bool bz=(j==i-1);
        while(j)
        {
            int k=j-1;
            while(k&&(p[i]-p[k])*(p[j]-p[k])>0)k--;
            double area=fabs((p[i]-O)*(p[j]-O))/2;
            if(k)area+=dp[j][k];
            if(bz)dp[i][j]=area;
            ans=max(ans,area),j=k;
        }
        if(bz)for(int j=1;j<i;j++)dp[i][j]=max(dp[i][j],dp[i][j-1]);
    }
}

int main()
{
    //freopen("lx.in","r",stdin);
    T=getint();
    while(T--)
    {
        n=getint();ans=0;
        for(int i=1;i<=n;i++)a[i].x=getint(),a[i].y=getint();
        for(int i=1;i<=n;i++)
        {
            O=a[i],m=0;
            for(int j=1;j<=n;j++)
                if(a[j].y>a[i].y||a[j].y==a[i].y&&a[j].x>a[i].x)p[++m]=a[j];
            solve();
        }
        printf("%0.1lf\n",ans);
    }
    return 0;
}
```

## 8.坐标轴

### 旋转坐标轴至新的向量上

```
#include <bits/stdc++.h>
#define EPS (1e-9)
using namespace std;
typedef long double ldb;

int X[3], Y[3], R[3];

int sgn(long double x) {
    if (x < -EPS) return -1;
    else if (x > EPS) return 1;
    else return 0;
}

// 将 x 轴旋转到向量 (xb, yb) 上，求出 (x, y) 旋转后对应的坐标
void rotate(ldb xb, ldb yb, ldb &x, ldb &y) {
    ldb d = sqrt(xb * xb + yb * yb);
    ldb xx = xb / d * x + yb / d * y;
    ldb yy = xb / d * y - yb / d * x;
    x = xx; y = yy;
}

void solve() {
    for (int i = 0; i < 3; i++) scanf("%d%d%d", &X[i], &Y[i], &R[i]);
    // 将 (x0, y0) 移到坐标原点
    ldb x1 = X[1] - X[0], y1 = Y[1] - Y[0];
    ldb x2 = X[2] - X[0], y2 = Y[2] - Y[0];
    ldb r0 = R[0], r1 = R[1], r2 = R[2];
    // 限制 r >= max(r0, r1, r2)
    ldb lim = max({(ldb) 0, r0, r1, r2});

    // 将 x 轴旋转到向量 (x1, y1) 上
    rotate(x1, y1, x2, y2);
    rotate(x1, y1, x1, y1);
    assert(sgn(y1) == 0);

    if (sgn(y2) == 0) {
        // 三个圆心共线
        ldb a = 2 * (r1 - r0) / x1 - 2 * (r2 - r0) / x2;
        ldb b = x1 + (r0 * r0 - r1 * r1) / x1 - x2 - (r0 * r0 - r2 * r2) / x2;
        if (sgn(a) == 0) {
            // 一次方程的一次项为 0，特判无解以及无数组解
            if (sgn(b) == 0) printf("-1\n");
            else printf("0\n");
        } else {
            // 一次方程的一次项不为 0，正常解方程
            ldb rs = -b / a;
            if (sgn(rs - lim) < 0) printf("0\n");
            else {
                // 计算 y^2 的值
                ldb xs = 2 * (r1 - r0) * rs + x1 * x1 - r1 * r1 + r0 * r0;
                xs /= 2 * x1;
                ldb ys2 = (rs - r0) * (rs - r0) - xs * xs;
                if (sgn(ys2) < 0) printf("0\n");
                else if (sgn(ys2) == 0) printf("1 %.12Lf\n", rs);
                else printf("2 %.12Lf\n", rs);
            }
        }
    } else {
        // 三个圆心不共线
        ldb a1 = -2 * x1;
        ldb c1 = 2 * (r1 - r0);
        ldb d1 = x1 * x1 - r1 * r1 + r0 * r0;
        ldb a2 = -2 * x2;
        ldb b2 = -2 * y2;
        ldb c2 = 2 * (r2 - r0);
        ldb d2 = x2 * x2 + y2 * y2 - r2 * r2 + r0 * r0;

        ldb p1 = -d1 / a1;
        ldb q1 = -c1 / a1;
        ldb p2 = (a2 * d1 - a1 * d2) / (a1 * b2);
        ldb q2 = (a2 * c1 - a1 * c2) / (a1 * b2);

        ldb a = q1 * q1 + q2 * q2 - 1;
        ldb b = (p1 * q1 + p2 * q2 + r0) * 2;
        ldb c = p1 * p1 + p2 * p2 - r0 * r0;

        // 二次方程所有系数都是 0，无数组解
        if (sgn(a) == 0 && sgn(b) == 0 && sgn(c) == 0) printf("-1\n");
        else if (sgn(a) == 0) {
            // 二次项为 0，实际上是一次方程
            if (sgn(b) == 0) printf("0\n");
            else {
                ldb rs = -c / b;
                if (sgn(rs - lim) < 0) printf("0\n");
                else printf("1 %.12Lf\n", rs);
            }
        } else {
            // 判别式法，解普通的二次方程
            ldb delta = b * b - 4 * a * c;
            if (sgn(delta) < 0) printf("0\n");
            else if (sgn(delta) == 0) {
                ldb rs = -b / (2 * a);
                if (sgn(rs - lim) < 0) printf("0\n");
                else printf("1 %.12Lf\n", rs);
            } else {
                ldb rs1 = (-b - sqrt(delta)) / (2 * a);
                ldb rs2 = (-b + sqrt(delta)) / (2 * a);
                if (rs1 > rs2) swap(rs1, rs2);
                if (sgn(rs1 - rs2) == 0) rs1 = -1;
                if (sgn(rs2 - lim) < 0) printf("0\n");
                else if (sgn(rs1 - lim) < 0) printf("1 %.12Lf\n", rs2);
                else printf("2 %.12Lf\n", rs1);
            }
        }
    }
}

int main() {
    int tcase; scanf("%d", &tcase);
    while (tcase--) solve();
    return 0;
}
```

