#include <cxxabi.h>
#include <execinfo.h>

#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <functional>
#include <type_traits>
#include <atomic>

namespace {

const char* kStrData = "abcd";

class MyString {
  public:
    MyString()
    :data_(nullptr)
    ,length_(0) {
      std::cout << "Call to default constructor" << std::endl;
    }

    MyString(const char* data) {
      std::cout << "Call to char* constructor" << std::endl;
      Clone(data);
    }

    MyString(const MyString& my_string) {
      std::cout << "Call to copy constructor" << std::endl;
      Clone(my_string.data_);
    }

    MyString(MyString&& my_string) noexcept {
      std::cout << "Call to move constructor" << std::endl;
      data_ = my_string.data_;
      length_ = my_string.length_;

      my_string.data_ = nullptr;
      my_string.length_ = 0;
    }

    ~MyString() {
      std::cout << "Call to destructor" << std::endl;
    }

    MyString& operator=(const MyString& my_string) {
      std::cout << "Call to operator =" << std::endl;
      Clone(my_string.data_);
      return *this;
    }

    MyString& operator=(MyString&& my_string) {
      if (&my_string != this) {
        std::cout << "Call to move operator =" << std::endl;
        data_ = my_string.data_;
        length_ = my_string.length_;

        my_string.data_ = nullptr;
        my_string.length_ = 0;
      }
      return *this;
    }

    void Show() {
      if (nullptr != data_) {
        std::cout << "MyString => " << data_ << std::endl;
      } else {
        std::cout << "MyString => null" << std::endl;
      }
    }

    std::string ToString() const {
      return (nullptr != data_ ? data_ : "");
    }

  private:
    void Clone(const char* data) {
      if (nullptr != data) {
        length_ = std::strlen(data) + 1;
        data_ = new char[length_];
        if (nullptr != data_) {
          std::copy_n(data, std::strlen(data), data_);
          data_[length_] = '\0';
        }
      }
    }

    void Move(MyString&& my_string) {
      data_ = my_string.data_;
      length_ = my_string.length_;

      my_string.data_ = nullptr;
      my_string.length_ = 0;
    }
  private:
    char* data_;
    size_t length_;
};

std::ostream& operator<<(std::ostream& os, const MyString& my_str) {
  os << my_str.ToString();
  return os;
}

MyString MakeMyString(const char* data) {
  MyString my_str(data);
#if 0
  return std::move(my_str);
#else
  return my_str;
#endif
}

std::unique_ptr<MyString> MakeMyStringPtr(const char* data) {
  std::unique_ptr<MyString> my_str(new MyString(data));
  return my_str;
}

void TestRightValue() {
  int i = 0;
  int& lr = i;
  const int& lr2 = 10;

#if 0
  MyString str1(kStrData);
  MyString str2 = str1;

  MyString str3;
  str3 = str1;
#endif

#if 0
   * 1. 如果没有禁用返回值优化，不管是否定义了move constructor, 都输出如下：
   * Call to char* constructor
   * MyString => abcd
   * Call to destructor
   * 
   * 2. 如果禁用返回值优化-fno-elide-constructors，输出如下：
   * Call to char* constructor
   * Call to move constructor MakeMyString中的局部变量move constructor到此函数的返回值临时变是中
   * Call to destructor
   * Call to move constructor
   * Call to destructor
   * MyString => abcd
   * Call to destructor
   *
   * 3. 如果没定义move constructor, 则上面调用的move constructor换成copy constructor

  MyString str4(MakeMyString(kStrData));
  str4.Show();
#endif

#if 0
  MyString str5;
  str5 = MakeMyString(kStrData);
  str5.Show();

  MyString str5(MakeMyString(kStrData));
  MyString&& str6 = std::move(str5);
  str6.Show();
  str5.Show();

  MyString str7(MakeMyString(kStrData));
#endif

  std::unique_ptr<MyString> str8 = MakeMyStringPtr(kStrData);
  str8->Show();
}

void Fun1(int& data) {
  std::cout << "Call to Fun1 with int" << std::endl;
}

void Fun1(int&& data) {
  std::cout << "Call to Fun1 with int&&" << std::endl;
}
//void Fun1()
template <class A>
void Fun2(A&& data) {
#if 0
  Fun1(data);
#else
  //完美转发的目标函数的参数，要么是&, 要么是&&
  Fun1(std::forward<A>(data));
#endif
}

void TestForward() {
  int data = 1;
  Fun2(data);
  Fun2(2);
}

void TestAuto() {
  int data = 0;
  int& rdata = data;
  auto& d1 = rdata;
  std::cout << "d1 type by typeid" << typeid(d1).name() << std::endl;
  std::cout << "data vs rdata : " << (typeid(data) == typeid(rdata) ? 1: 0) << std::endl;
}

template <class ContainerT>
class Foo {
  //typedef typename ContainerT::iterator ItType;
  //使用decltype, 对于const ContainerT和非const的同时适用
  typedef decltype(ContainerT().begin()) ItType;
public:
  void ShowFirst(ContainerT& container) {
    ItType it = container.begin();
    std::cout << "Container first:" << *it << std::endl;
  }
};

void TestDecltype() {
  typedef const std::vector<int> IntVectorType;
  Foo<IntVectorType> foo;
  IntVectorType int_vec = {1, 2, 3};
  foo.ShowFirst(int_vec);
}

using MyStrVector = std::vector<MyString>;
using MyFunPtr = void(*)(int);

void TestRangeFor() {
#if 1
//以下这两种方式都先调用char*构造函数，再调用copy constructor
#if 0
  MyStrVector mystr_vector = {
    "abc",
    "def",
    "glm"
  };
#else
  MyStrVector mystr_vector = {
    MyString("abc"),
    MyString("def"),
    MyString("glm"),
  };
#endif
#else
//以下这种方式都先调用char*构造函数，再调用move constructor
  MyStrVector mystr_vector;
  mystr_vector.push_back(MyString("abc"));
  mystr_vector.push_back(MyString("def"));
  mystr_vector.push_back(MyString("glm"));
#endif

  for (const auto& str : mystr_vector) {
    std::cout << str << std::endl;
  }
}

struct FuncObjT {
  void operator()(int data) {
    std::cout << "Call to FuncObjT::() | data:" << data << std::endl;
  }
};

void TestStdFunction() {
  FuncObjT func_obj;
  //func_obj(100);
  std::function<void(int)> fun = func_obj;
  fun(100);

  using FunPtrT = void(*)(int);
  FunPtrT fun_ptr = [](int) {
  };
}

template <class T>
std::string type_name() {
  typedef typename std::remove_reference<T>::type TR;

  std::unique_ptr<char, void(*)(void*)> own(
#ifndef __GNUC__
     nullptr,
#else
     abi::__cxa_demangle(typeid(TR).name(), nullptr,
       nullptr, nullptr),
#endif
     std::free);

  std::string r = (own != nullptr ? own.get() : typeid(TR).name());

  if (std::is_const<TR>::value)
    r += " const";
  if (std::is_volatile<TR>::value)
    r += " volatile";
  if (std::is_lvalue_reference<T>::value)
    r += "&";
  else if (std::is_rvalue_reference<T>::value)
    r += "&&";

  return r;
}

void TestValueType() {
  int i = 0;
  auto&& rv = i;
  std::cout << "rv type:" << type_name<int>() << std::endl;
}

void TestStringMove() {
  std::string source_str("abc");
  std::cout << "source str: " << source_str << std::endl;

  std::string dest_str = std::move(source_str);
  std::cout << "source str: " << source_str << std::endl;
  std::cout << "dest str: " << dest_str << std::endl;

  std::vector<std::string> str_vec;
  str_vec.emplace(str_vec.end(), std::move(dest_str));
  std::cout << "dest str: " << dest_str << std::endl;
}

// __func__支持用在构造函数中
class TestFoo {
  public:
    TestFoo()
      :class_name_(__func__) {
      }

    const char* Name() const {
      return class_name_;
    }
  private:
    const char* class_name_;
};

void Test__func__() {
  TestFoo tf;
  std::cout << tf.Name() << std::endl;
}

// noexcept表明当前函数不会抛出异常，
// 编译时会给出下面的warning:
//t_cc.cc:335:9: warning: throw will always call terminate() [-Wterminate]
//   throw 1;
void FuncWithNoexcept() noexcept(false) {
  throw 1;
}

void TestNoexcept() noexcept {
  try {
    TestNoexcept();
  } catch (...) {
    std::cout << "sss" << std::endl;
  }
}

struct TFA {
  int i {
    0
  };
};

class T1 {
  public:
    virtual void Show() = 0;
};

class T2 : public T1 {
  public:
    virtual void Show() /*final*/ {
      std::cout << "Call to  T2::Show" << std::endl;
    }
};

class T3 : public T2 {
  public:
    virtual void Show() {
    }
};

void TestSizeofWithLong() {
  std::cout << sizeof(short int) << std::endl;//2
  std::cout << sizeof(int) << std::endl;//4
  std::cout << sizeof(long int) << std::endl;//8
  std::cout << sizeof(long long int) << std::endl;//8
  std::cout << sizeof(float) << std::endl;//4
  std::cout << sizeof(double) << std::endl;//8
}

int ReturnRValue() {
  int tmp = 10;
  return int(10);
}

void TestRValue() {
  int&& d = ReturnRValue();
  std::cout << "RValue" << d << std::endl;

  std::cout << std::boolalpha;

  std::cout << std::is_lvalue_reference<decltype("sss")>::value << std::endl;

  std::cout << std::is_lvalue_reference<int&>::value << std::endl;
  std::cout << std::is_lvalue_reference<int>::value << std::endl;

  int i = 0;
  std::cout << std::is_rvalue_reference<decltype(i++)>::value << std::endl;
  std::cout << std::is_lvalue_reference<decltype(++i)>::value << std::endl;
}

constexpr void TestConstexpr() {
  bool flag = true;
  if (flag) {
    return;
  } else {
    return ;
  }
}

void TestAtomicWithLockfree() {
  struct A { int a[100]; };
  struct B { int x, y; };

  std::cout << std::boolalpha
    << "std::atomic<A> lock free? "
    //如果加上下面这句，链接时需要 -latomic
    << std::atomic<A>{}.is_lock_free() //false
    << std::endl
    << "std::atomic<B> lock free? "
    << std::atomic<B>{}.is_lock_free()//true
    << std::endl;
}

template<typename T>
void Add(T&& t) {
}

template<typename T>
class TFoo {
  public:
    //模块类的成员函数不能实现完美转发，必须要定义成模板
    template<typename U, typename = std::enable_if<std::is_same<T, U>::value>>
    void Add(U&& u) {

    }
};

/*
template<typename T>
void TFoo<T>::Add(T&& t) {
}
*/

void TestShardPtr() {
  std::shared_ptr<int> data(new int(10));
  Add(data);

  TFoo<std::shared_ptr<int>> tf;
  std::shared_ptr<std::string> data1(new std::string(""));
  tf.Add(data1);
}

struct TTT {
  //这个写在类内部，则这个TTT是trivial类型
  //写在类外部，则不是
  TTT() = default;
  ~TTT() = delete;
  int a;
};
//TTT::TTT() = default;
void TestTrivial() {
  std::cout << std::is_trivial<TTT>::value << std::endl;
  std::cout << std::is_pod<TTT>::value << std::endl;
}

class TF {
  public:
    TF() {
        std::cout << "Call to TF default constructor" << std::endl;
    }

    TF(const TF& tf)
      :data_(tf.data_) {
        std::cout << "Call to TF copy constructor" << std::endl;
      }

    TF(TF&& tf)
      :data_(std::move(tf.data_)) {
        std::cout << "Call to TF move constructor" << std::endl;
      }

    TF(std::string&& data)
      :data_(data) {
        std::cout << "Call to TF std::string constructor" << std::endl;
      }
  private:
    std::string data_;
};

class TFR {
  public:
    TFR() {
        std::cout << "Call to TFR default constructor" << std::endl;
    }

    //这里tfr是个右值引用，但实际上它是左值
    TFR(TFR&& tfr)
      :data_(std::move(tfr.data_)) {
      //:data_(tfr.data_) {
        std::cout << "Call to TFR move constructor" << std::endl;
      }

    TFR(std::string&& data)
      :data_(std::move(data)) {
        std::cout << std::is_rvalue_reference<decltype(data)>::value << std::endl;
        std::cout << "Call to TFR std::string constructor" << std::endl;
      }
  private:
    //std::string data_;
    TF data_;
};

void TestTFR() {
  TFR tfr1;
  TFR tfr2 = std::move(tfr1);
  std::string data("sss");
  TFR tfr3(std::string("sss"));
}

struct SSS {
  char a;
  int b;
};
struct align {
  int    a;
  char   b;
  short  c;
  char d;
};// __attribute((aligned(8)));

struct testttt {
  char a;
  char b;
  double c;
  int d;
};

struct __attribute__ ((aligned(8))) my_struct1 {
  char a;
  char b;
  char c;
  short f[3];
};//sizeof = 8

//1)将结构体内所有数据成员的长度值相加，记为sum_a； 
//2)将各数据成员为了内存对齐，按各自对齐模数而填充的字节数累加到和sum_a上，记为sum_b。对齐模数是#pragma pack指定的数值以及该数据成员自身长度中数值较小者。该数据相对起始位置应该是对齐模式的整数倍; 
//3)将和sum_b向结构体模数对齐，该模数是【#pragma pack指定的数值】、【未指定#pragma pack时，系统默认的对齐模数（32位系统为4字节，64位为8字节）】和【结构体内部最大的基本数据类型成员】长度中数值较小者。结构体的长度应该是该模数的整数倍。  
//(4)__attribute__(aligned(8))的对齐规和#pragma pack一样，只是结构体的总大小必须是 aligned(n)中n的整数倍   
void TestAlign() {
  std::cout << sizeof(my_struct1) << std::endl;//8
  std::cout << sizeof(SSS) << std::endl;//8

  printf("__attribute((align)): %zd\n"
      "                     a: %p\n"
      "                     b: %p\n"
      "                     c: %p\n"
      "                     d: %p\n",
      sizeof(struct align),//16
      (&((struct align*)0)->a),//0
      (&((struct align*)0)->b),//4
      (&((struct align*)0)->c),//6
      (&((struct align*)0)->d));//8

  printf("__attribute((testttt)): %zd\n"
      "                     a: %p\n"
      "                     b: %p\n"
      "                     c: %p\n"
      "                     d: %p\n",
      sizeof(struct testttt),//16
      (&((struct testttt*)0)->a),//0
      (&((struct testttt*)0)->b),//4
      (&((struct testttt*)0)->c),//6
      (&((struct testttt*)0)->d));//8
}
} //namespace 

int main(int argc, char* argv[]) {
  //TestRightValue();
  //TestForward();
  //TestAuto();
  //TestDecltype();
  //TestRangeFor();
  //TestStdFunction();
  //TestValueType();
  //TestStringMove();
  //TestNoexcept();
  //TestSizeofWithLong();
  //TestRValue();
  //TestAtomicWithLockfree();
  //TestTFR();
  TestAlign();

  return 0;
}
