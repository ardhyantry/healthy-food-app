@extends('layouts.adminSB')

@section('title', 'Quickbite - Menu List')

@section('content')
    <div class="container">
        <h2>Daftar Menu</h2>
        <a href="{{ route('admin.menus.create') }}" class="btn btn-success mb-3">Tambah Menu</a>

        @if (session('success'))
            <div class="alert alert-success alert-dismissible fade show" role="alert">
                {{ session('success') }}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Tutup"></button>
            </div>
        @endif

        <table class="table table-bordered">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Nama</th>
                    <th>Nutrition Facts</th>
                    <th>Deskripsi</th>
                    <th>Harga</th>
                    <th>Kategori</th>
                    <th>Aksi</th>
                </tr>
            </thead>
            <tbody>
                @foreach ($listMenu as $m)
                    <tr>
                        <td>{{ $m->id }}</td>
                        <td>{{ $m->name }}</td>
                        <td>{{ $m->nutrition_fact }}</td>
                        <td>{{ $m->description }}</td>
                        <td>{{ number_format($m->price, 0, ',', '.') }}</td>
                        <td>{{ $m->category->name }}</td>
                        <td>
                            <a href="{{ route('admin.menus.show', $m->id) }}" class="btn btn-info btn-sm">View</a>
                            <a href="#modalEdit" class="btn btn-primary btn-sm" data-bs-toggle="modal"
                                onclick="getEditForm({{ $m->id }})">
                                Edit
                            </a>
                            <button class="btn btn-danger btn-sm" data-bs-toggle="modal" data-bs-target="#deleteModal{{ $m->id }}">
                                Delete
                            </button>
                        </td>
                    </tr>

                    <!-- Delete Modal -->
                    <div class="modal fade" id="deleteModal{{ $m->id }}" tabindex="-1" aria-labelledby="deleteModalLabel{{ $m->id }}"
                        aria-hidden="true">
                        <div class="modal-dialog">
                            <div class="modal-content">
                                <div class="modal-header bg-danger text-white">
                                    <h5 class="modal-title" id="deleteModalLabel{{ $m->id }}">Konfirmasi Hapus</h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Tutup"></button>
                                </div>
                                <div class="modal-body">
                                    Apakah kamu yakin ingin menghapus menu <strong>{{ $m->name }}</strong>?
                                </div>
                                <div class="modal-footer">
                                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Batal</button>
                                    <form action="{{ route('admin.menus.destroy', $m->id) }}" method="POST">
                                        @csrf
                                        @method('DELETE')
                                        <button type="submit" class="btn btn-danger">Hapus</button>
                                    </form>
                                </div>
                            </div>
                        </div>
                    </div>
                @endforeach
            </tbody>
            <tfoot>
                <tr>
                    <td colspan="7" class="text-center">Total Menu: {{ $listMenu->count() }}</td>
                </tr>
            </tfoot>
        </table>
    </div>
    @push('modals')
        <div class="modal fade" id="modalEdit" tabindex="-1" role="basic" aria-hidden="true">
            <div class="modal-dialog modal-wide">
                <div class="modal-content">
                    <div class="modal-body" id="modalContent">
                        {{-- You can put animated loading image here... --}}
                        <div class="d-flex justify-content-center align-items-center" style="height: 200px;">
                            <div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                        </div>

                    </div>
                </div>
            </div>
        </div>
    @endpush
    @push('scripts')
        <script>
            function getEditForm(id) {
                $.ajax({
                    type: 'POST',
                    url: '{{route("admin.menus.getEditForm")}}',
                    data: {
                        '_token': '<?php echo csrf_token() ?>',
                        'id': id
                    },
                    success: function (data) {
                        $('#modalContent').html(data.msg)
                    }
                });
            }
        </script>
    @endpush
@endsection
