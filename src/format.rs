use std::fmt;

pub const RULE_WIDTH: usize = 97;

pub trait Pretty: Copy {
    fn fmt_pretty(self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

fn write_thousands(f: &mut fmt::Formatter<'_>, n: usize) -> fmt::Result {
    if n >= 1000 {
        write_thousands(f, n / 1000)?;
        write!(f, ",{:03}", n % 1000)
    } else {
        write!(f, "{n}")
    }
}

impl Pretty for usize {
    fn fmt_pretty(self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self >= 1_000_000_000 {
            write!(f, "{:.3e}", self as f64)
        } else {
            write_thousands(f, self)
        }
    }
}

impl Pretty for f64 {
    fn fmt_pretty(self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if !self.is_finite() {
            return write!(f, "{self}");
        }
        let abs = self.abs();
        if abs != 0.0 && !(1e-3..1e6).contains(&abs) {
            write!(f, "{self:.3e}")
        } else if abs >= 1_000.0 {
            if self.is_sign_negative() {
                f.write_str("-")?;
            }
            write_thousands(f, abs.round() as usize)
        } else {
            write!(f, "{self:.6}")
        }
    }
}

impl Pretty for f32 {
    fn fmt_pretty(self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self as f64).fmt_pretty(f)
    }
}

pub fn pretty(v: impl Pretty) -> impl fmt::Display {
    fmt::from_fn(move |f| v.fmt_pretty(f))
}

pub fn write_rule(f: &mut fmt::Formatter<'_>, ch: &str) -> fmt::Result {
    for _ in 0..RULE_WIDTH {
        f.write_str(ch)?;
    }
    Ok(())
}

pub fn print_rule(ch: &str) {
    println!("{}", fmt::from_fn(|f| write_rule(f, ch)));
}

pub fn write_banner(f: &mut fmt::Formatter<'_>, title: &str) -> fmt::Result {
    write_rule(f, "=")?;
    writeln!(f)?;
    writeln!(f, "{:^width$}", title, width = RULE_WIDTH)?;
    write_rule(f, "=")?;
    writeln!(f)
}

pub fn begin_section(
    f: &mut fmt::Formatter<'_>,
    title: &str,
    has_section: &mut bool,
) -> fmt::Result {
    if *has_section {
        write_rule(f, "-")?;
        writeln!(f)?;
    }
    *has_section = true;
    writeln!(f, "  {title}")
}

pub fn write_field(
    f: &mut fmt::Formatter<'_>,
    label: &str,
    value: impl fmt::Display,
) -> fmt::Result {
    writeln!(f, "    {label:<14}: {value}")
}
