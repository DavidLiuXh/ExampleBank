#include <cstring>
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

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

MyString MakeMyString(const char* data) {
  MyString my_str(data);
#if 0
  return std::move(my_str);
#else
  return my_str;
#endif
}

std::unique_ptr<MyString>&& MakeMyStringPtr(const char* data) {
  std::unique_ptr<MyString> my_str(new MyString(data));
  return std::move(my_str);
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

} //namespace 

int main(int argc, char* argv[]) {
  //TestRightValue();
  //TestForward();
  //TestAuto();
  TestDecltype();

  return 0;
}
