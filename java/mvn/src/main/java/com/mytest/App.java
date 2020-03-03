package com.mytest;

import java.util.Objects;

/**
 * Hello world!
 *
 */
public class App {

    public static final class CaseInsensitiveString {
        private final String s;

        public CaseInsensitiveString(String s) {
            this.s = Objects.requireNonNull(s);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                System.out.println("o == this");
                return true;
            } else if (o instanceof CaseInsensitiveString) {
                return s.equalsIgnoreCase(
                        ((CaseInsensitiveString) o).s);
            } else {
                return false;
            }
        }

        public void show() {
            System.out.println("Call to show func");
        }
    }

    public static void main( String[] args ) {
        System.out.println( "Hello World!" );
        CaseInsensitiveString cis = new CaseInsensitiveString("Polish");
        String s = "polish";
        System.out.println(cis.toString());
        System.out.println(cis.equals(s));
        System.out.println(cis.equals(cis));
        System.out.println(s.equals(cis));

        Animal animal = Animal.create("dog");
        System.out.println(animal.name());
        System.out.println(animal.hashCode());
        System.out.println(animal.toString());
    }
}
